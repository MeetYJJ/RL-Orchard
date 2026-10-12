#!/usr/bin/env python3
"""Single-process launch manager for the Full Pick V2 ROS graph.

The manager owns only processes that it starts.  Existing healthy ROS nodes are
reused, while a stale process with no ROS graph entry is reported instead of
being duplicated.  Full Pick keeps its own hardware gate and reads the
operator's ENABLE_HARDWARE/START_FULL_PICK prompts from the terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "runtime.yaml"
STOP_SERVICE = "/rm_json_bridge/stop"
STOP_SERVICE_TYPE = "std_srvs/srv/Trigger"
ARM_STATE_SERVICE = "/rm_json_bridge/get_arm_state"
ARM_STATE_SERVICE_TYPE = "orchard_picker/srv/GetArmState"


class StartupError(RuntimeError):
    pass


def _scalar(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        if (len(value) >= 2 and value[0] == value[-1] and
                value[0] in "\"'"):
            return value[1:-1]
        return value


def _fallback_yaml(text: str) -> Dict[str, Any]:
    """Parse the small mapping-only runtime config without PyYAML."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if ":" not in content:
            raise ValueError("unsupported runtime.yaml line: {}".format(raw))
        key, raw_value = content.split(":", 1)
        key = key.strip().strip("\"'")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value.strip():
            value: Dict[str, Any] = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _scalar(raw_value)
    return root


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        value = yaml.safe_load(text)
    except ImportError:
        value = _fallback_yaml(text)
    if not isinstance(value, dict):
        raise ValueError("runtime config must be a mapping")
    return value


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_file: Any
    started_by_manager: bool = True
    output_thread: Optional[threading.Thread] = None


class ProcessManager:
    def __init__(self, config_path: Path = DEFAULT_CONFIG,
                 command_runner: Optional[Callable[..., Any]] = None):
        self.config_path = Path(config_path).resolve()
        document = load_config(self.config_path)
        self.config = document["launch_manager"]["ros__parameters"]
        self.vision_parameters = document["apple_detector"]["ros__parameters"]
        self.root = Path(self.config.get("project_root", ROOT))
        self.setup_script = Path(self.config["setup_script"])
        self.log_dir = Path(self.config.get("log_dir", "/tmp/rl_full_pick_launch"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ros_log_dir = Path(self.config.get("ros_log_dir", self.log_dir / "ros_log"))
        self.ros_home = Path(self.config.get("ros_home", self.log_dir / "ros_home"))
        self.ros_log_dir.mkdir(parents=True, exist_ok=True)
        self.ros_home.mkdir(parents=True, exist_ok=True)
        self.discovery_timeout = float(self.config.get("discovery_timeout_s", 20.0))
        self.poll_interval = float(self.config.get("poll_interval_s", 0.25))
        self.env = os.environ.copy()
        self.env.update({
            "RL_ORCHARD_SKIP_RL_OVERLAY": "1",
            "ROS_LOG_DIR": str(self.ros_log_dir),
            "ROS_HOME": str(self.ros_home),
        })
        self.command_runner = command_runner
        self.processes: Dict[str, ManagedProcess] = {}
        self.external: Dict[str, bool] = {}
        self.events: "queue.Queue[str]" = queue.Queue()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._environment_ready = False
        self._ready_deadline: Optional[float] = None
        self._last_graph: Dict[str, Any] = {}
        self._ready_markers: Set[str] = set()
        self._bridge_ready_confirmed = False
        self._pick_task_manager_ready_confirmed = False
        self._tf_components_confirmed = False

    def log(self, message: str) -> None:
        line = str(message)
        self.events.put(line)
        print(line, flush=True)

    def _ros_shell(self, argv: Sequence[str]) -> str:
        setup = shlex.quote(str(self.setup_script))
        command = shlex.join([str(item) for item in argv])
        exports = "export RL_ORCHARD_SKIP_RL_OVERLAY=1 ROS_LOG_DIR={} ROS_HOME={}".format(
            shlex.quote(str(self.ros_log_dir)), shlex.quote(str(self.ros_home))
        )
        return "source {} >/dev/null && {} && exec {}".format(setup, exports, command)

    def _prepare_environment(self) -> None:
        """Source once; every query and managed process gets the same overlay.

        Environment initialization is outside the DDS discovery deadline. In
        particular, its latency must not consume each 3-second graph query.
        """
        if self._environment_ready or self.command_runner is not None:
            return
        result = subprocess.run(
            ["/bin/bash", "-lc", self._ros_shell(["/usr/bin/env", "-0"])],
            cwd=str(self.root), env=self.env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15.0,
        )
        if result.returncode:
            raise StartupError("ROS_ENVIRONMENT_FAILED: " + result.stderr.decode(errors="replace"))
        self.env = {
            key: value for entry in result.stdout.decode(errors="surrogateescape").split("\0")
            if "=" in entry for key, value in [entry.split("=", 1)]
        }
        for key in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "PYTHONPATH"):
            self.env[key] = ":".join(
                path for path in self.env.get(key, "").split(":")
                if path and not any(path == str(self.root / directory) or
                                    path.startswith(str(self.root / directory) + "/")
                                    for directory in ("install", "build"))
            )
        self.env["PYTHONPATH"] = str(self.root) + ":" + self.env.get("PYTHONPATH", "")
        # Preserve the caller's DDS scope.  Forcing localhost-only here makes
        # an already-running bridge disappear from this process's graph.
        self._environment_ready = True

    def _run_ros(self, argv: Sequence[str], timeout: float = 3.0):
        self._prepare_environment()
        if self.command_runner is not None:
            return self.command_runner(list(argv), timeout=timeout)
        return subprocess.run(
            list(argv),
            cwd=str(self.root), env=self.env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )

    @staticmethod
    def _lines(output: str) -> Set[str]:
        return {line.strip() for line in output.splitlines() if line.strip()}

    def _graph(self, fields: Sequence[str] = ("nodes", "services", "topics")) -> Dict[str, Any]:
        def typed(lines: str) -> Dict[str, str]:
            result: Dict[str, str] = {}
            for line in lines.splitlines():
                match = re.match(r"^\s*(\S+)\s+\[([^]]+)\]", line)
                if match:
                    result[match.group(1)] = match.group(2)
            return result

        def names_or_typed(lines: str) -> Dict[str, str]:
            """Read both Foxy ``topic list`` formats.

            Topic type discovery can lag behind the graph itself.  Readiness
            only needs topic names, so an untyped line such as
            ``/apple/target_point`` is deliberately accepted as well as the
            typed ``/apple/target_point [geometry_msgs/msg/PointStamped]``
            form.
            """
            result: Dict[str, str] = {}
            for line in lines.splitlines():
                line = line.strip()
                if not line:
                    continue
                name = line.split()[0]
                if name.startswith("/"):
                    match = re.match(r"^(\S+)\s+\[([^]]+)\]", line)
                    result[name] = match.group(2) if match else ""
            return result

        graph: Dict[str, Any] = {"nodes": set(), "services": {}, "topics": {}, "error": ""}
        commands = {
            "nodes": ["ros2", "node", "list", "--no-daemon"],
            "services": ["ros2", "service", "list", "-t", "--no-daemon"],
            # Do not request topic types here.  Foxy can expose a topic name
            # before its type metadata is available, and type readiness is not
            # part of the detector startup contract.
            "topics": ["ros2", "topic", "list", "--no-daemon"],
        }
        errors = []
        for field in fields:
            try:
                # DDS graph queries can take several seconds on Foxy while a
                # camera node is creating publishers.  Keep each query's own
                # bounded timeout independent of the outer readiness deadline;
                # shrinking it to the remaining poll time caused the final
                # topic query to be aborted before it could return.
                result = self._run_ros(commands[field], timeout=5.0)
                if result.returncode != 0:
                    raise StartupError("{} query failed: {}".format(field, result.stdout))
                output = result.stdout or ""
                if field == "nodes":
                    graph[field] = self._lines(output)
                elif field == "topics":
                    graph[field] = names_or_typed(output)
                else:
                    graph[field] = typed(output)
            except (OSError, subprocess.SubprocessError, StartupError) as exc:
                errors.append("{}: {}".format(field, exc))
        graph["error"] = "; ".join(errors)
        self._last_graph = graph
        return graph

    @staticmethod
    def _has_node(nodes: Set[str], name: str) -> bool:
        return name in nodes or name.lstrip("/") in nodes

    def _log_vision_debug(self, node_found: bool, positions_found: bool,
                          target_found: bool, ready: bool,
                          type_unknown: bool) -> None:
        """Print one complete readiness sample.

        These values come from the same ``--no-daemon`` graph snapshots used
        for the decision.  Printing every sample makes a delayed Foxy graph
        observable during the bounded readiness poll.
        """
        self.log("VISION_NODE_FOUND={}".format(str(bool(node_found)).lower()))
        self.log("VISION_POSITIONS_TOPIC_FOUND={}".format(
            str(bool(positions_found)).lower()))
        self.log("VISION_TARGET_TOPIC_FOUND={}".format(
            str(bool(target_found)).lower()))
        self.log("VISION_TOPIC_TYPE_UNKNOWN={}".format(
            str(bool(type_unknown)).lower()))
        self.log("VISION_READY_FINAL={}".format(str(bool(ready)).lower()))

    def _bridge_ready(self, graph: Optional[Dict[str, Any]] = None) -> bool:
        # Observe nodes and services as one readiness sample.  Separate Foxy
        # CLI samples can be temporarily inconsistent during discovery.
        graph = graph if graph is not None else self._graph(("nodes", "services"))
        bridge = self.config["bridge"]
        if not self._has_node(graph["nodes"], bridge["node"]):
            return False
        if graph.get("error"):
            return False
        self._last_graph = graph
        ready = all(graph["services"].get(name) == service_type
                    for name, service_type in bridge["services"].items())
        if ready and "bridge_services" not in self._ready_markers:
            for name, service_type in bridge["services"].items():
                self.log("SERVICE_READY {} = true (type={})".format(
                    name, service_type))
            self._ready_markers.add("bridge_services")
        if ready:
            self._bridge_ready_confirmed = True
        return ready

    def _bridge_process_healthy(self) -> bool:
        """Use the successful service handshake for the final preflight.

        The Bridge may be external to this manager, so a fresh CLI graph
        query is not a reliable liveness signal on Foxy.  It was already
        required to expose all four typed services before startup continued.
        For an owned Bridge we additionally require its process to remain
        alive; an external Bridge is left untouched and trusted only after
        that successful handshake.
        """
        if not self._bridge_ready_confirmed:
            return False
        item = self.processes.get("bridge")
        if item is not None and item.process.poll() is not None:
            self._tail_log("bridge")
            return False
        self.log("BRIDGE_PROCESS_HEALTHY=true")
        return True

    @staticmethod
    def _topics_ready(graph: Dict[str, Any], required: Sequence[str]) -> bool:
        return all(name in graph.get("topics", {}) for name in required)

    def _pick_task_manager_ready(self, graph: Optional[Dict[str, Any]] = None) -> bool:
        """Confirm the task-manager data path is alive.

        On Foxy, a CLI ``node list`` sample can be empty while the same
        participant's topic graph already contains the task-manager
        publishers.  Treating that transient node snapshot as proof that the
        process is absent caused the launcher to tear down a healthy manager
        (and, in an SSH session, made the session appear to disconnect).  The
        strict path still requires the node plus both topics.  The fallback
        requires both topics *and* a live process owned by this manager (or a
        matching external process), so a stale topic name alone can never
        satisfy readiness.
        """
        graph = graph if graph is not None else self._graph(("nodes", "topics"))
        spec = self.config.get("pick_task_manager", {})
        node_name = spec.get("node", "/pick_task_manager")
        node_found = self._has_node(graph["nodes"], node_name)
        if (node_found and not graph["topics"] and
                "topics:" not in graph.get("error", "")):
            topics_graph = self._graph(("topics",))
            graph["topics"] = topics_graph["topics"]
            graph["error"] = topics_graph["error"]
        self._last_graph = graph
        required = tuple(spec.get("topics", {
            "/pick_task_manager/state": "*",
            "/pick_task_manager/target_base": "*",
        }).keys())
        topics_found = self._topics_ready(graph, required)
        # A Foxy graph entry can outlive the Python process that created it.
        # Require a live owned or external process before accepting node/topic
        # names as readiness evidence; otherwise a stale participant can make
        # the launcher skip the required restart.
        owned = self.processes.get("pick_task_manager")
        owned_alive = bool(owned is not None and owned.process.poll() is None)
        external_alive = self._external_process_alive(r"[p]ick_task_manager\.py")
        if node_found and topics_found and (owned_alive or external_alive):
            return True
        if not topics_found:
            return False

        if owned_alive or external_alive:
            self.log("PICK_TASK_MANAGER_READY_FROM_TOPICS=true")
            self.log("PICK_TASK_MANAGER_NODE_GRAPH_TRANSIENT=true")
            return True
        return False

    def _pick_task_manager_process_healthy(self) -> bool:
        if not self._pick_task_manager_ready_confirmed:
            return False
        item = self.processes.get("pick_task_manager")
        if item is not None and item.process.poll() is not None:
            self._tail_log("pick_task_manager")
            return False
        if item is None and not self._external_process_alive(r"[p]ick_task_manager\.py"):
            # The graph may still contain stale task-manager names after the
            # process exited.  Do not let that stale graph unblock Full Pick.
            return False
        self.log("PICK_TASK_MANAGER_PROCESS_HEALTHY=true")
        return True

    def _full_pick_ready(self, graph: Optional[Dict[str, Any]] = None) -> bool:
        """Require the Full Pick node and its own READY_FULL_PICK marker."""
        graph = graph if graph is not None else self._graph(("nodes",))
        node_name = self.config.get("full_pick", {}).get("node", "/full_pick_v2")
        if not self._has_node(graph["nodes"], node_name):
            return False
        # An owned process must complete its own ROS/service/feedback precheck
        # and print the established READY_FULL_PICK marker.  Node discovery
        # alone is insufficient because a process can create its node and then
        # fail on UDP or service precheck immediately afterwards.
        if "full_pick" in self.processes:
            return "full_pick" in self._ready_markers
        return True

    def _tf_ready(self, graph: Optional[Dict[str, Any]] = None) -> bool:
        """Check the complete TF publisher chain used by the picker.

        The node checks ensure that the dynamic flange publisher and both
        static calibration publishers are alive.  The topic checks ensure
        that their TF broadcasters have created the standard ROS TF channels;
        no robot command or TF service call is made here.
        """
        graph = graph if graph is not None else self._graph(("nodes", "topics"))
        spec = self.config.get("tf", {})
        node_names = tuple(spec.get("nodes", {
            "tcp_tf_publisher": "/tcp_tf_publisher",
            "hand_eye_static_tf_publisher": "/hand_eye_static_tf_publisher",
            "gripper_tcp_static_tf_publisher": "/gripper_tcp_static_tf_publisher",
        }).values())
        nodes_found = all(self._has_node(graph["nodes"], name)
                          for name in node_names)
        if (nodes_found and not graph["topics"] and
                "topics:" not in graph.get("error", "")):
            topics_graph = self._graph(("topics",))
            graph["topics"] = topics_graph["topics"]
            graph["error"] = topics_graph["error"]
        required_topics = tuple(spec.get("topics", ("/tf", "/tf_static")))
        topics_found = nodes_found and self._topics_ready(graph, required_topics)
        self._last_graph = graph
        self.log("TF_NODES_FOUND={}".format(str(nodes_found).lower()))
        self.log("TF_TOPICS_FOUND={}".format(str(topics_found).lower()))
        self.log("TF_READY_FINAL={}".format(
            str(bool(nodes_found and topics_found)).lower()))
        return nodes_found and topics_found

    def _tf_node_ready(self, node_name: str) -> bool:
        graph = self._graph(("nodes",))
        return self._has_node(graph["nodes"], node_name)

    def _tf_processes_healthy(self) -> bool:
        """Check TF processes after the initial graph readiness handshake.

        Foxy can return an incomplete graph from a fresh CLI participant even
        while the already-started broadcasters are still alive.  The initial
        ``start_tf`` wait has already observed all TF nodes and topics; the
        Full Pick preflight therefore uses process liveness for this final
        check instead of treating one transient graph sample as a failure.
        """
        names = tuple(self.config.get("tf", {}).get("nodes", {
            "tcp_tf_publisher": "/tcp_tf_publisher",
            "hand_eye_static_tf_publisher": "/hand_eye_static_tf_publisher",
            "gripper_tcp_static_tf_publisher": "/gripper_tcp_static_tf_publisher",
        }).keys())
        for name in names:
            item = self.processes.get(name)
            if item is not None and item.process.poll() is not None:
                self._tail_log(name)
                return False
        static_markers = {
            "hand_eye_static_tf_publisher": "Published static hand-eye TF on /tf_static",
            "gripper_tcp_static_tf_publisher": "Published static TF on /tf_static",
        }
        for name, marker in static_markers.items():
            if name in self.processes and not self._log_contains(name, (marker,)):
                return False
        self.log("TF_PROCESS_HEALTHY=true")
        return True

    def _vision_ready(self, graph: Optional[Dict[str, Any]] = None) -> bool:
        graph = graph if graph is not None else self._graph(("nodes", "topics"))
        vision = self.config["vision"]
        node_found = self._has_node(graph["nodes"], vision["node"])
        if (node_found and not graph["topics"] and
                "topics:" not in graph.get("error", "")):
            topics_graph = self._graph(("topics",))
            graph["topics"] = topics_graph["topics"]
            graph["error"] = topics_graph["error"]
        self._last_graph = graph
        # Only graph presence is required.  In particular, do not call
        # ``topic info`` or compare topic types: those metadata queries can
        # lag in ROS 2 Foxy even while the detector is already publishing.
        positions_found = node_found and "/apple_positions" in graph["topics"]
        target_found = node_found and "/apple/target_point" in graph["topics"]
        topics_found = positions_found and target_found
        # A plain ``ros2 topic list --no-daemon`` line has no type metadata.
        # That is diagnostic only; it must never block detector readiness.
        type_unknown = any(
            not str(graph["topics"].get(name, "")).strip()
            for name, found in (("/apple_positions", positions_found),
                                ("/apple/target_point", target_found))
            if found
        )
        ready = node_found and topics_found
        self._log_vision_debug(node_found, positions_found, target_found,
                               ready, type_unknown)
        if not ready:
            # A fresh Foxy CLI participant can miss a running camera node and
            # its publishers for several seconds.  For a detector started by
            # this manager, accept the detector's own completed initialization
            # markers only while its process is still alive.  This is stronger
            # than process existence alone and lets Full Pick perform its own
            # subscriber/precheck handshake.
            item = self.processes.get("vision")
            if (item is not None and item.process.poll() is None and
                    self._log_contains("vision", (
                        "Apple detector started; publishing on /apple_positions",
                        "Publishing picker target point on /apple/target_point",
                    ))):
                self.log("VISION_READY_FROM_PROCESS_LOG=true")
                return True
        return ready

    def _external_process_alive(self, pattern: str) -> bool:
        try:
            result = subprocess.run(["pgrep", "-f", pattern],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except OSError:
            return False

    def wait_ready(self, name: str, probe: Callable[[], bool], timeout: Optional[float] = None) -> bool:
        self._prepare_environment()
        deadline = time.monotonic() + (self.discovery_timeout if timeout is None else timeout)
        last_reason = ""
        self._ready_deadline = deadline
        try:
            while time.monotonic() < deadline:
                if self._shutdown_done:
                    raise StartupError("STARTUP_CANCELLED_DURING_SHUTDOWN")
                item = self.processes.get(name)
                if item is not None and item.process.poll() is not None:
                    self._tail_log(name)
                    raise StartupError("{}_PROCESS_EXITED = {}".format(name.upper(), item.process.returncode))
                try:
                    # The probe itself is bounded.  If it started before the
                    # deadline and returns a valid ready result just after
                    # it, keep that successful observation instead of
                    # discarding it because the final monotonic sample
                    # crossed the deadline by a few milliseconds.
                    if probe():
                        self.log("{}_READY=true".format(name.upper()))
                        return True
                except Exception as exc:
                    last_reason = repr(exc)
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    # A polling interval, never an unconditional startup sleep.
                    time.sleep(min(self.poll_interval, remaining))
        finally:
            self._ready_deadline = None
        if last_reason:
            self.log("{}_READY_ERROR = {}".format(name.upper(), last_reason))
        return False

    def start_process(self, name: str, argv: Sequence[str], *, inherit_stdin: bool = False,
                      stream_output: bool = False) -> ManagedProcess:
        if self._shutdown_done:
            raise StartupError("STARTUP_CANCELLED_DURING_SHUTDOWN")
        self._prepare_environment()
        log_path = self.log_dir / (name + ".log")
        # A new launch gets a fresh log.  Appending stale output can make a
        # failed software-only precheck look as if this invocation connected
        # to the robot or started an old node.
        log_file = log_path.open("wb")
        stdout: Any = subprocess.PIPE if stream_output else log_file
        try:
            process = subprocess.Popen(
                list(argv), cwd=str(self.root), env=self.env,
                stdin=None if inherit_stdin else subprocess.DEVNULL,
                stdout=stdout, stderr=subprocess.STDOUT,
                text=stream_output, bufsize=1 if stream_output else -1,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise
        item = ManagedProcess(name, process, log_file)
        self.processes[name] = item
        self._ready_markers.discard(name)
        (self.log_dir / (name + ".pid")).write_text(str(process.pid), encoding="ascii")
        self.log("{}_PROCESS = started; PID = {}".format(name.upper(), process.pid))
        if stream_output:
            item.output_thread = threading.Thread(
                target=self._forward_output, args=(item,), daemon=True,
            )
            item.output_thread.start()
        return item

    def _forward_output(self, item: ManagedProcess) -> None:
        stream = item.process.stdout
        if stream is None:
            return
        pending = ""
        try:
            # ``input()`` writes its prompt without a newline.  A readline
            # forwarder would therefore hide the operator prompt forever and
            # make the child appear hung.  Forward small chunks and flush;
            # this still keeps all child output out of the manager's stdin.
            while True:
                chunk = stream.read(1)
                if not chunk:
                    break
                pending += chunk
                if item.name == "full_pick" and "READY_FULL_PICK" in pending:
                    self._ready_markers.add("full_pick")
                item.log_file.write(chunk.encode("utf-8", errors="replace"))
                item.log_file.flush()
                print(chunk, end="", flush=True)
                if len(pending) > 4096:
                    pending = pending[-1024:]
        finally:
            stream.close()

    def is_alive(self, name: str) -> bool:
        item = self.processes.get(name)
        return bool(item is not None and item.process.poll() is None)

    def _tail_log(self, name: str, lines: int = 20) -> None:
        path = self.log_dir / (name + ".log")
        if not path.exists():
            return
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        except OSError:
            return
        if tail:
            self.log("{}_LOG_TAIL = {}".format(name.upper(), " | ".join(tail)))

    def _log_contains(self, name: str, fragments: Sequence[str]) -> bool:
        """Return true only when this invocation's process logged all markers."""
        path = self.log_dir / (name + ".log")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return all(fragment in text for fragment in fragments)

    def _wait_or_fail(self, name: str, probe: Callable[[], bool]) -> None:
        if not self.wait_ready(name, probe):
            self._tail_log(name)
            graph = self._last_graph or self._graph()
            if graph.get("error"):
                self.log("{}_GRAPH_QUERY_ERROR = {}".format(name.upper(), graph["error"]))
            self.log("{}_MISSING_DEPENDENCY = true".format(name.upper()))
            self.log("{}_NODES = {}".format(name.upper(), sorted(graph["nodes"])))
            self.log("{}_SERVICES = {}".format(name.upper(), sorted(graph["services"])))
            self.log("{}_TOPICS = {}".format(name.upper(), sorted(graph["topics"])))
            raise StartupError("{}_READY_TIMEOUT".format(name.upper()))

    def _software_bridge_params(self, source: Path) -> Path:
        """Create a minimal ROS params overlay which disables TCP auto-connect.

        The source picker configuration remains untouched.  ``driver_only``
        still launches the real bridge node and advertises its normal service
        API, but this launch manager's default integration check cannot open a
        socket to a robot that is not part of the test.
        """
        if not source.is_file():
            raise StartupError("BRIDGE_PARAMS_MISSING = {}".format(source))
        destination = self.log_dir / "picker_params_software_only.yaml"
        content = (
            "rm_json_bridge:\n"
            "  ros__parameters:\n"
            "    auto_connect: false\n"
            "    set_tool_voltage_on_start: false\n"
        )
        try:
            if not destination.exists() or destination.read_text(encoding="utf-8") != content:
                destination.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise StartupError("BRIDGE_SOFTWARE_PARAMS_FAILED = {}".format(exc))
        return destination

    def start_bridge(self) -> None:
        self._prepare_environment()
        if self._bridge_ready():
            self.external["bridge"] = True
            self.log("BRIDGE_PROCESS = existing")
            self.log("BRIDGE_READY=true")
            return
        if (self._has_node(self._last_graph.get("nodes", set()), self.config["bridge"]["node"]) or
                self._external_process_alive(r"[r]m_json_bridge\.py|[r]os2 launch orchard_picker driver_only\.launch\.py")):
            self._wait_or_fail("bridge", lambda: self._bridge_ready())
            self.external["bridge"] = True
            self.log("BRIDGE_PROCESS = existing")
            return
        bridge = self.config["bridge"]
        command = [str(x) for x in self.config.get(
            "bridge_command",
            ["ros2", "launch", bridge["package"], bridge["launch"]],
        )]
        params_path = Path(bridge["params"])
        if bool(self.config.get("software_only", True)):
            # Keep the real bridge node and all service types available for
            # graph/precheck integration, but do not open the RealMan socket
            # during this software-only launcher run.
            params_path = self._software_bridge_params(params_path)
            self.log("BRIDGE_SOFTWARE_ONLY=true")
        self.start_process("bridge", command + [
            "config:=" + str(params_path),
        ])
        self._wait_or_fail("bridge", lambda: self._bridge_ready())

    def start_tf(self) -> None:
        """Start the GUI/README-validated TF publishers after the bridge."""
        self._prepare_environment()
        if self._tf_ready():
            self.external["tf"] = True
            self.log("TF_PROCESS = existing")
            self.log("TF_READY=true")
            return

        spec = self.config.get("tf", {})
        params = Path(spec.get(
            "params", "/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml"
        ))
        if not params.is_file():
            raise StartupError("TF_PARAMS_MISSING = {}".format(params))
        commands = self.config.get("tf_commands", {
            "tcp_tf_publisher": ["ros2", "run", "orchard_picker", "tcp_tf_publisher.py"],
            "hand_eye_static_tf_publisher": [
                "ros2", "run", "orchard_picker", "hand_eye_static_tf_publisher.py"
            ],
            "gripper_tcp_static_tf_publisher": [
                "ros2", "run", "orchard_picker", "gripper_tcp_static_tf_publisher.py"
            ],
        })
        # Use the project copies for these two pure ROS communication nodes so
        # the static-TF fix is effective even when the ROS2 workspace contains
        # an older installed package.  The dynamic TCP publisher remains the
        # GUI-validated orchard_picker executable.
        project_python = Path(self.config.get("python", sys.executable))
        project_scripts = {
            "hand_eye_static_tf_publisher": self.root / "src" / "orchard_picker" /
            "scripts" / "hand_eye_static_tf_publisher.py",
            "gripper_tcp_static_tf_publisher": self.root / "src" / "orchard_picker" /
            "scripts" / "gripper_tcp_static_tf_publisher.py",
        }
        nodes = spec.get("nodes", {
            "tcp_tf_publisher": "/tcp_tf_publisher",
            "hand_eye_static_tf_publisher": "/hand_eye_static_tf_publisher",
            "gripper_tcp_static_tf_publisher": "/gripper_tcp_static_tf_publisher",
        })
        patterns = {
            "tcp_tf_publisher": r"[t]cp_tf_publisher\.py",
            "hand_eye_static_tf_publisher": r"[h]and_eye_static_tf_publisher\.py",
            "gripper_tcp_static_tf_publisher": r"[g]ripper_tcp_static_tf_publisher\.py",
        }
        for process_name, node_name in nodes.items():
            if self._tf_node_ready(node_name):
                self.external[process_name] = True
                self.log("{}_PROCESS = existing".format(process_name.upper()))
                continue
            if (self._has_node(self._last_graph.get("nodes", set()), node_name) or
                    self._external_process_alive(patterns.get(process_name, process_name))):
                self._wait_or_fail(process_name,
                                   lambda node_name=node_name: self._tf_node_ready(node_name))
                self.external[process_name] = True
                self.log("{}_PROCESS = existing".format(process_name.upper()))
                continue
            command = [str(x) for x in commands.get(process_name, [])]
            local_script = project_scripts.get(process_name)
            if local_script is not None and local_script.is_file():
                command = [str(project_python), str(local_script)]
                self.log("{}_COMMAND_SOURCE = project".format(process_name.upper()))
            if not command:
                raise StartupError("TF_COMMAND_MISSING = {}".format(process_name))
            self.start_process(process_name, command + [
                "--ros-args", "--params-file", str(params),
            ])
            self._wait_or_fail(process_name,
                               lambda node_name=node_name: self._tf_node_ready(node_name))

        # Each broadcaster has already passed its own node readiness wait.
        # Requiring a fresh all-nodes graph snapshot here is unreliable on
        # Foxy (it may contain only one of the three nodes).  Confirm the
        # owned processes and the static publishers' actual /tf_static log.
        self._tf_components_confirmed = True
        self._wait_or_fail("tf", lambda: self._tf_processes_healthy())
        self.log("TF_READY_FROM_COMPONENTS=true")

    def start_vision(self) -> None:
        if self._vision_ready():
            self.external["vision"] = True
            self.log("VISION_PROCESS = existing")
            self.log("VISION_READY=true")
            return
        if (self._has_node(self._last_graph.get("nodes", set()), self.config["vision"]["node"]) or
                self._external_process_alive(r"[a]pple_detector|[r]un_rl_vision\.sh")):
            self._wait_or_fail("vision", lambda: self._vision_ready())
            self.external["vision"] = True
            self.log("VISION_PROCESS = existing")
            return
        vision = self.config["vision"]
        command = [str(x) for x in self.config.get(
            "vision_command", ["ros2", "run", vision["package"], vision["executable"]],
        )]
        model_path = self.vision_parameters.get("yolo_model_path", "")
        if not model_path or not Path(model_path).is_file():
            raise StartupError("VISION_YOLO_WEIGHTS_MISSING = " + str(model_path))
        self.start_process("vision", command + [
            "--ros-args", "--params-file", str(self.config_path),
        ])
        self._wait_or_fail("vision", lambda: self._vision_ready())
        self.log("VISION_PARAMETER_LOAD_PASS = true")

    def start_pick_task_manager(self) -> None:
        """Start the traditional target/state adapter after vision is ready."""
        self._prepare_environment()
        if self._pick_task_manager_ready():
            self.external["pick_task_manager"] = True
            self._pick_task_manager_ready_confirmed = True
            self.log("PICK_TASK_MANAGER_PROCESS = existing")
            self.log("PICK_TASK_MANAGER_READY=true")
            return

        spec = self.config.get("pick_task_manager", {})
        node_name = spec.get("node", "/pick_task_manager")
        if self._external_process_alive(r"[p]ick_task_manager\.py"):
            self._wait_or_fail("pick_task_manager",
                               lambda: self._pick_task_manager_ready())
            self._pick_task_manager_ready_confirmed = True
            self.external["pick_task_manager"] = True
            self.log("PICK_TASK_MANAGER_PROCESS = existing")
            self.log("PICK_TASK_MANAGER_READY=true")
            return

        command = [str(x) for x in self.config.get(
            "pick_task_manager_command",
            spec.get("command", ["ros2", "run", "orchard_picker",
                                  "pick_task_manager.py"]),
        )]
        params = Path(spec.get(
            "params", "/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml"
        ))
        if not params.is_file():
            raise StartupError("PICK_TASK_MANAGER_PARAMS_MISSING = {}".format(params))
        # This launch path is a software/precheck path.  Keep the traditional
        # node and its topics available, while preventing an accidental task
        # service call from issuing robot motion during startup validation.
        self.start_process("pick_task_manager", command + [
            "--ros-args", "--params-file", str(params),
            "-p", "dry_run:=true",
        ])
        self._wait_or_fail("pick_task_manager",
                           lambda: self._pick_task_manager_ready())
        self._pick_task_manager_ready_confirmed = True

    def start_full_pick(self) -> None:
        # Dependencies may be briefly absent from one Foxy graph sample even
        # after their own startup wait succeeded.  Recheck each dependency
        # using the same bounded polling contract instead of failing on a
        # single transient snapshot.
        dependencies = (
            ("bridge", self._bridge_process_healthy, "FULL_PICK_BLOCKED_BRIDGE_NOT_READY"),
            ("tf", self._tf_processes_healthy, "FULL_PICK_BLOCKED_TF_NOT_READY"),
            ("vision", self._vision_ready, "FULL_PICK_BLOCKED_VISION_NOT_READY"),
            ("pick_task_manager", self._pick_task_manager_process_healthy,
             "FULL_PICK_BLOCKED_PICK_TASK_MANAGER_NOT_READY"),
        )
        for dependency, probe, reason in dependencies:
            if not self.wait_ready(dependency, probe):
                raise StartupError(reason)
        python_bin = Path(self.config.get("python", sys.executable))
        if not python_bin.exists():
            raise StartupError("FULL_PICK_PYTHON_MISSING = {}".format(python_bin))
        if bool(self.config["full_pick"].get("hardware", False)):
            raise StartupError("FULL_PICK_HARDWARE_CONFIG_MUST_BE_FALSE")
        self.log("FULL_PICK_HARDWARE_DEFAULT = false")
        command = [str(python_bin)] + [str(x) for x in self.config["full_pick"]["command"]]
        if self._full_pick_ready():
            self.external["full_pick"] = True
            self.log("FULL_PICK_PROCESS = existing")
            self.log("FULL_PICK_READY=true")
            return
        self.start_process("full_pick", command, inherit_stdin=True, stream_output=True)
        self._wait_or_fail("full_pick", lambda: self._full_pick_ready())
        self.log("FULL_PICK_PROCESS = started")

    def start_all(self) -> bool:
        try:
            self.start_bridge()
            self.start_tf()
            self.start_vision()
            self.start_pick_task_manager()
            self.start_full_pick()
            self.log("STARTUP_MANAGER_READY = true")
            return True
        except Exception as exc:
            self.log("STARTUP_MANAGER_FAILED = {}".format(exc))
            self.shutdown()
            return False

    def _service_call(self, service: str, service_type: str, request: str,
                      timeout: float = 5.0) -> bool:
        try:
            result = self._run_ros([
                "ros2", "service", "call", service, service_type, request,
            ], timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log("SERVICE_CALL_ERROR = {}".format(exc))
            return False
        output = result.stdout or ""
        self.log("SERVICE_CALL {} = {}".format(service, output.strip() or result.returncode))
        return result.returncode == 0

    def _stop_robot(self) -> None:
        graph = self._graph()
        if graph["services"].get(STOP_SERVICE) != STOP_SERVICE_TYPE:
            self.log("ROBOT_STOP_FEEDBACK = unavailable_stop_service")
            return
        sent = self._service_call(STOP_SERVICE, STOP_SERVICE_TYPE, "{}", timeout=5.0)
        if not sent:
            self.log("ROBOT_STOP_FEEDBACK = stop_request_failed")
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._service_call(ARM_STATE_SERVICE, ARM_STATE_SERVICE_TYPE, "{}", timeout=2.0):
                self.log("ROBOT_STOP_FEEDBACK = observed")
                return
            time.sleep(self.poll_interval)
        self.log("ROBOT_STOP_FEEDBACK = unconfirmed")

    def stop_process(self, name: str, timeout: float = 5.0) -> None:
        item = self.processes.pop(name, None)
        if item is None:
            return
        process = item.process
        if process.poll() is None:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if process.poll() is not None:
                    break
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + timeout
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if process.poll() is not None:
                    break
                timeout = 2.0
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if item.output_thread is not None:
            item.output_thread.join(timeout=1.0)
        item.log_file.close()
        try:
            (self.log_dir / (name + ".pid")).unlink()
        except FileNotFoundError:
            pass
        self.log("{}_PROCESS_STOPPED = true".format(name.upper()))

    def _wait_owned_nodes_gone(self, names: Sequence[str]) -> bool:
        if not names:
            return True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            graph = self._graph(("nodes",))
            # An empty graph together with a query error is not proof that
            # owned nodes exited; retain the leak warning in that case.
            if not graph.get("error") and all(
                    not self._has_node(graph["nodes"], name) for name in names):
                return True
            time.sleep(self.poll_interval)
        return False

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        # The order is intentional: no Full Pick input remains after its
        # process is fenced, then the controller is asked to stop, then the
        # camera and bridge processes owned by this manager are released.
        owned_nodes = []
        if "vision" in self.processes:
            owned_nodes.append(self.config["vision"]["node"])
        if "pick_task_manager" in self.processes:
            owned_nodes.append(self.config.get("pick_task_manager", {}).get(
                "node", "/pick_task_manager"
            ))
        tf_nodes = self.config.get("tf", {}).get("nodes", {
            "tcp_tf_publisher": "/tcp_tf_publisher",
            "hand_eye_static_tf_publisher": "/hand_eye_static_tf_publisher",
            "gripper_tcp_static_tf_publisher": "/gripper_tcp_static_tf_publisher",
        })
        for process_name, node_name in tf_nodes.items():
            if process_name in self.processes:
                owned_nodes.append(node_name)
        if "bridge" in self.processes:
            owned_nodes.append(self.config["bridge"]["node"])
        self.stop_process("full_pick")
        self.stop_process("pick_task_manager")
        for process_name in tf_nodes:
            self.stop_process(process_name)
        # Never send a cleanup stop to a bridge that this invocation did not
        # start.  An external bridge may belong to another application.
        if "bridge" in self.processes and self._bridge_ready():
            if bool(self.config.get("software_only", True)):
                self.log("ROBOT_STOP_SKIPPED_SOFTWARE_ONLY=true")
            else:
                self._stop_robot()
        self.stop_process("vision")
        self.stop_process("bridge")
        if self._wait_owned_nodes_gone(owned_nodes):
            self.log("NO_ROS_PROCESS_LEAK = true")
        else:
            self.log("NO_ROS_PROCESS_LEAK = false")
        self.log("PROCESS_CLEANUP_COMPLETE = true")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch Full Pick V2 ROS dependencies")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    manager = ProcessManager(Path(args.config))

    def handle_signal(_signum, _frame):
        manager.log("SHUTDOWN_SIGNAL_RECEIVED = true")
        manager.shutdown()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if not manager.start_all():
        return 2
    try:
        process = manager.processes.get("full_pick")
        if process is None:
            # Full Pick may have been an already-running healthy external
            # node.  Keep the manager alive in that case instead of exiting
            # immediately and tearing down the dependencies it just checked.
            while manager._full_pick_ready():
                time.sleep(0.25)
            return 0
        while process.process.poll() is None:
            time.sleep(0.25)
        return 0 if process.process.returncode == 0 else int(process.process.returncode)
    except KeyboardInterrupt:
        return 130
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
