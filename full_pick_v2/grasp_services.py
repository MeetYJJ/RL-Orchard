"""ROS service types from the authoritative workspace; one-shot mutations."""
from __future__ import annotations
import importlib
from pathlib import Path
import time
from .safety import PickFault

SERVICE_PACKAGE = Path("/home/nvidia/ros2_ws/install/orchard_picker/lib/python3.8/site-packages/orchard_picker")
SERVICE_SPECS = (
    ("get_arm_state", "GetArmState"),
    ("gripper_pick", "GripperPick"),
    ("gripper_release", "GripperRelease"),
    ("movej", "MoveJ"),
)

def import_service_types():
    package = importlib.import_module("orchard_picker")
    package.__path__ = [str(SERVICE_PACKAGE)] + [p for p in package.__path__ if p != str(SERVICE_PACKAGE)]
    services = importlib.import_module("orchard_picker.srv")
    if SERVICE_PACKAGE.resolve() not in Path(services.__file__).resolve().parents:
        raise PickFault(f"ROS_SERVICE_PACKAGE_SHADOWED:{services.__file__}")
    types = {name: getattr(services, name) for name in ("GetArmState","GripperPick","GripperRelease","MoveJ")}
    from std_srvs.srv import Trigger
    types["Trigger"] = Trigger
    for cls in types.values():
        type(cls).__import_type_support__()
        if type(cls)._TYPE_SUPPORT is None:
            raise PickFault(f"ROS_TYPE_SUPPORT_MISSING:{cls.__name__}")
    return types

class GraspServices:
    def __init__(self, node, config, spin):
        self.node = node
        self.types = import_service_types()
        self.config, self.spin = config, spin
        self.clients = {}
        self.client_errors = {}
        for name, cls in SERVICE_SPECS + (("stop", "Trigger"),):
            service_name = "/rm_json_bridge/" + name
            try:
                self.clients[name] = node.create_client(self.types[cls], service_name)
            except Exception as exc:
                # Keep the error attached to the individual service.  A failed
                # client must never be collapsed into a generic PRECHECK error.
                self.clients[name] = None
                self.client_errors[name] = repr(exc)
        self.sent = set()
        self._last_report = {}

    @staticmethod
    def _type_name(cls):
        return "orchard_picker/srv/" + cls.__name__

    def _graph(self):
        try:
            entries = self.node.get_service_names_and_types()
            return {name: tuple(types) for name, types in entries}, None
        except BaseException as exc:
            return {}, repr(exc)

    def diagnose(self, wait_s=0.0, emit=True):
        """Inspect this node's graph and clients without calling a service.

        The graph and all readiness checks intentionally use the same Node and
        Context that will later be used for the one-shot calls.  This avoids
        treating a cached ``ros2 service list`` result as proof that this
        process can reach a server.
        """
        graph, graph_error = self._graph()
        report = {}
        for name, cls_name in SERVICE_SPECS:
            full_name = "/rm_json_bridge/" + name
            expected_type = self._type_name(self.types[cls_name])
            visible_types = graph.get(full_name, ())
            visible = bool(visible_types)
            type_match = expected_type in visible_types
            client = self.clients.get(name)
            created = client is not None
            error = self.client_errors.get(name)
            ready_before = False
            wait_result = False
            elapsed_ms = 0.0
            if client is not None:
                try:
                    ready_before = bool(client.service_is_ready())
                    if not ready_before and wait_s > 0.0:
                        started = time.monotonic()
                        wait_result = bool(client.wait_for_service(timeout_sec=float(wait_s)))
                        elapsed_ms = (time.monotonic() - started) * 1000.0
                    else:
                        wait_result = ready_before
                    ready = bool(client.service_is_ready())
                except Exception as exc:
                    ready = False
                    error = repr(exc)
            else:
                ready = False
            report[name] = {
                "name": full_name,
                "type": expected_type,
                "visible": visible,
                "visible_types": visible_types,
                "type_match": type_match,
                "client_created": created,
                "ready": ready,
                "wait_result": wait_result,
                "elapsed_ms": elapsed_ms,
                "error": error,
            }
        self._last_report = report
        if emit:
            if graph_error:
                print("PYTHON_RCLPY_SERVICE_GRAPH_ERROR = " + graph_error, flush=True)
            print("PYTHON_RCLPY_SERVICES = " + repr(sorted(graph)), flush=True)
            print("PYTHON_RCLPY_GRAPH_OK = " + str(all(item["visible"] for item in report.values())).lower(), flush=True)
            print("SERVICE_TYPES_MATCH = " + str(all(item["type_match"] for item in report.values())).lower(), flush=True)
            print("CLIENT_CREATION_FIXED = " + str(all(item["client_created"] for item in report.values())).lower(), flush=True)
            for name, _ in SERVICE_SPECS:
                item = report[name]
                print("SERVICE_NAME = " + item["name"], flush=True)
                print("SERVICE_TYPE = " + item["type"], flush=True)
                print("SERVICE_VISIBLE_IN_GRAPH = " + str(item["visible"]).lower(), flush=True)
                print("SERVICE_TYPE_MATCH = " + str(item["type_match"]).lower(), flush=True)
                if item["visible_types"] and not item["type_match"]:
                    print("SERVICE_GRAPH_TYPES = " + repr(list(item["visible_types"])), flush=True)
                print("CLIENT_CREATED = " + str(item["client_created"]).lower(), flush=True)
                print("SERVICE_IS_READY = " + str(item["ready"]).lower(), flush=True)
                print("WAIT_FOR_SERVICE_RESULT = " + str(item["wait_result"]).lower(), flush=True)
                print("WAIT_ELAPSED_MS = " + f'{item["elapsed_ms"]:.1f}', flush=True)
                print("SERVICE_ERROR = " + (item["error"] or "none"), flush=True)
                print("PYTHON_SERVICE_READY " + name + " = " + str(item["ready"]).lower(), flush=True)
        return report

    def wait_for_required(self, wait_s=0.2):
        """Wait briefly on each required client and return a real bool."""
        report = self.diagnose(wait_s=wait_s, emit=False)
        return all(report[name]["ready"] for name, _ in SERVICE_SPECS)

    def _is_ready(self, name):
        client = self.clients.get(name)
        if client is None:
            return False
        try:
            return bool(client.service_is_ready())
        except Exception as exc:
            self.client_errors.setdefault(name, repr(exc))
            return False

    def ready(self):
        return all(
            self._is_ready(name)
            for name, _ in SERVICE_SPECS
        )

    def missing(self):
        return [
            "/rm_json_bridge/" + name
            for name, _ in SERVICE_SPECS
            if not self._is_ready(name)
        ]

    def call(self, name, request, timeout, once=True, monitor=None):
        if once and name in self.sent:
            raise PickFault("DUPLICATE_SERVICE_BLOCKED:"+name)
        client = self.clients[name]
        if client is None or not self._is_ready(name):
            raise PickFault("SERVICE_UNAVAILABLE:"+name)
        # Mark before submitting: an exception/timeout has unknown outcome.
        if once:
            self.sent.add(name)
        future = client.call_async(request)
        deadline = time.monotonic()+timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise PickFault("SERVICE_RESULT_UNKNOWN_TIMEOUT:"+name)
            self.spin(.01)
            if monitor is not None:
                monitor()
        result = future.result()
        if result is None:
            raise PickFault("SERVICE_RESULT_UNKNOWN:"+name)
        if not result.success:
            raise PickFault(f"SERVICE_FAILED:{name}:{result.message}")
        return result

    def arm_state(self):
        result = self.call("get_arm_state", self.types["GetArmState"].Request(), 8.0, once=False)
        if result.err:
            raise PickFault(f"CONTROLLER_ERROR:service:{result.err}")
        return result

    def close_gripper(self):
        req = self.types["GripperPick"].Request()
        req.speed, req.force, req.wait = self.config.pick_speed, self.config.pick_force, True
        print("GRIPPER_CLOSE_REQUESTED", flush=True)
        try:
            result = self.call("gripper_pick", req, 30.)
        except BaseException as exc:
            print(f"GRIPPER_CLOSE_SERVICE_RESULT = {exc}", flush=True)
            print("GRIPPER_CLOSE_SENT = "+str("gripper_pick" in self.sent).lower(), flush=True)
            raise
        print(f"GRIPPER_CLOSE_SERVICE_RESULT = success:{result.message}", flush=True)
        print("GRIPPER_CLOSE_SENT = true", flush=True)

    def home(self, monitor):
        req = self.types["MoveJ"].Request()
        req.joint_deg = self.config.home_joint_deg.tolist()
        req.speed, req.blend_radius, req.wait = 10, 0, True
        self.call("movej", req, self.config.service_timeout, monitor=monitor)

    def release(self):
        req = self.types["GripperRelease"].Request()
        req.speed, req.wait = self.config.release_speed, True
        self.call("gripper_release", req, 30.)

    def stop(self):
        req = self.types["Trigger"].Request()
        return self.call("stop", req, 5.)
