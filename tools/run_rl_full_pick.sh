#!/usr/bin/env bash
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the authoritative Foxy + ros2_ws graph. RL-Orchard's collection
# overlay has a shadow orchard_picker package and is intentionally excluded.
export RL_ORCHARD_SKIP_RL_OVERLAY=1
source "$project_dir/scripts/setup_rl_env.sh"
set -u
cd "$project_dir"

strip_rl_overlay() {
  local value="$1" entry result=""
  local IFS=:
  for entry in $value; do
    [[ "$entry" == "$project_dir/install"* || "$entry" == "$project_dir/build"* ]] && continue
    [[ -n "$entry" ]] || continue
    if [[ -n "$result" ]]; then result+=":$entry"; else result="$entry"; fi
  done
  printf '%s' "$result"
}
export AMENT_PREFIX_PATH="$(strip_rl_overlay "${AMENT_PREFIX_PATH:-}")"
export CMAKE_PREFIX_PATH="$(strip_rl_overlay "${CMAKE_PREFIX_PATH:-}")"
export PYTHONPATH="$(strip_rl_overlay "${PYTHONPATH:-}")"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/rl_full_pick_v2/ros_log}"
export ROS_HOME="${ROS_HOME:-/tmp/rl_full_pick_v2/ros_home}"
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME" /tmp/rl_full_pick_v2

python_bin="${FULL_PICK_PYTHON:-/home/nvidia/miniconda3/envs/visionyolo/bin/python}"
ros2_bin="${FULL_PICK_ROS2:-ros2}"
vision_script="${FULL_PICK_VISION_SCRIPT:-$project_dir/scripts/run_rl_vision.sh}"
if [[ ! -x "$python_bin" ]]; then
  echo "FULL_PICK_PYTHON_MISSING = $python_bin" >&2
  exit 2
fi

check_only=false
for arg in "$@"; do
  [[ "$arg" == "--check-only" ]] && check_only=true
done
if [[ "$check_only" == true ]]; then
  exec "$python_bin" -m full_pick_v2 "$@"
fi

started_bridge=""
started_vision=""
bridge_process="existing"
vision_process="existing"

cleanup() {
  trap - INT TERM EXIT
  local pid
  for pid in "$started_vision" "$started_bridge"; do
    [[ -n "$pid" ]] || continue
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep .1
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

service_snapshot=""
node_snapshot=""
vision_snapshot=""

refresh_graph() {
  service_snapshot="$(timeout 2s "$ros2_bin" service list 2>/dev/null || true)"
  node_snapshot="$(timeout 2s "$ros2_bin" node list 2>/dev/null || true)"
}
services_ready() {
  local service
  for service in /rm_json_bridge/get_arm_state /rm_json_bridge/gripper_pick \
      /rm_json_bridge/gripper_release /rm_json_bridge/movej; do
    grep -qx "$service" <<<"$service_snapshot" || return 1
  done
  return 0
}
bridge_node_present() {
  grep -Eq '^/rm_json_bridge$|^rm_json_bridge$' <<<"$node_snapshot"
}
bridge_process_present() {
  pgrep -f '[r]m_json_bridge.py|[r]os2 launch orchard_picker driver_only.launch.py' >/dev/null 2>&1
}
vision_publisher_ready() {
  vision_snapshot="$(timeout 2s "$ros2_bin" topic info /apple/target_point 2>/dev/null || true)"
  grep -Eq 'Publisher count: [1-9][0-9]*' <<<"$vision_snapshot"
}
vision_node_present() {
  grep -Eq '^/apple_detector$|^apple_detector$' <<<"$node_snapshot"
}
vision_process_present() {
  pgrep -f '[a]pple_detector|run_rl_vision.sh' >/dev/null 2>&1
}

# Bridge first. Do not start a second bridge when a node is already visible;
# a graph mismatch is reported after the bounded discovery window.
bridge_attempts="${FULL_PICK_DISCOVERY_ATTEMPTS:-40}"
for _ in $(seq 1 "$bridge_attempts"); do
  refresh_graph
  if services_ready; then
    break
  fi
  if [[ -z "$started_bridge" ]] && ! bridge_node_present && ! bridge_process_present; then
    echo "BRIDGE_PROCESS = starting" >&2
    setsid "$ros2_bin" launch orchard_picker driver_only.launch.py \
      config:=/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml \
      </dev/null >/tmp/rl_full_pick_v2/bridge.log 2>&1 &
    started_bridge=$!
    bridge_process="started"
  fi
  sleep .5
done
refresh_graph
if ! services_ready; then
  echo "BRIDGE_PROCESS = $bridge_process" >&2
  for service in /rm_json_bridge/get_arm_state /rm_json_bridge/gripper_pick \
      /rm_json_bridge/gripper_release /rm_json_bridge/movej; do
    if grep -qx "$service" <<<"$service_snapshot"; then
      echo "SERVICE_READY $service = true" >&2
    else
      echo "SERVICE_READY $service = false" >&2
    fi
  done
  if (bridge_node_present || bridge_process_present) && [[ -z "$started_bridge" ]]; then
    echo "BRIDGE_GRAPH_NODE_PRESENT_BUT_SERVICES_MISSING = true" >&2
  fi
  [[ -f /tmp/rl_full_pick_v2/bridge.log ]] && tail -n 50 /tmp/rl_full_pick_v2/bridge.log >&2 || true
  exit 2
fi
echo "BRIDGE_PROCESS = $bridge_process" >&2
for service in get_arm_state gripper_pick gripper_release movej; do
  echo "SERVICE_READY $service = true" >&2
done

# Vision second. A visible apple_detector node without a publisher is reused
# and allowed to finish DDS discovery; it is never duplicated.
vision_attempts="${FULL_PICK_DISCOVERY_ATTEMPTS:-40}"
for _ in $(seq 1 "$vision_attempts"); do
  refresh_graph
  if vision_publisher_ready; then
    break
  fi
  if [[ -z "$started_vision" ]] && ! vision_node_present && ! vision_process_present; then
    echo "VISION_PROCESS = starting" >&2
    CAMERA_SERIAL="${CAMERA_SERIAL:-908212070350}" SHOW_IMAGE=false \
      setsid bash "$vision_script" \
      </dev/null >/tmp/rl_full_pick_v2/vision.log 2>&1 &
    started_vision=$!
    vision_process="started"
  fi
  sleep .5
done
refresh_graph
vision_publisher_ready || true
echo "VISION_PROCESS = $vision_process" >&2
if vision_publisher_ready; then
  echo "VISION_PUBLISHER_READY = true" >&2
  echo "ROS_ENVIRONMENT_MATCH = true" >&2
else
  echo "VISION_PUBLISHER_READY = false" >&2
  (vision_node_present || vision_process_present) && echo "VISION_NODE_PRESENT_BUT_PUBLISHER_MISSING = true" >&2 || true
  [[ -f /tmp/rl_full_pick_v2/vision.log ]] && tail -n 50 /tmp/rl_full_pick_v2/vision.log >&2 || true
  exit 2
fi

# The Python process inherits exactly this sanitized ROS graph environment.
set +e
"$python_bin" -m full_pick_v2 "$@"
status=$?
set -e
exit "$status"
