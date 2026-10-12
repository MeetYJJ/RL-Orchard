#!/usr/bin/env bash
set -eo pipefail

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source "$RL_ORCHARD_ROOT/scripts/setup_rl_env.sh"
set -u

ORCHARD_SHOW_IMAGE="${SHOW_IMAGE:-false}"
ORCHARD_CAMERA_SERIAL="${CAMERA_SERIAL:-}"
ORCHARD_CONFIG="$RL_ORCHARD_ROOT/src/apple_vision/config/apple_detector.yaml"

orchard_ros_args=(
  --ros-args
  --params-file "$ORCHARD_CONFIG"
  -p "show_image:=$ORCHARD_SHOW_IMAGE"
)
if [[ -n "$ORCHARD_CAMERA_SERIAL" ]]; then
  orchard_ros_args+=( -p "camera_serial:=$ORCHARD_CAMERA_SERIAL" )
fi

orchard_vision_pid=""
orchard_cleanup() {
  local attempt
  trap - HUP INT TERM EXIT
  if [[ -z "$orchard_vision_pid" ]]; then
    return
  fi
  kill -INT -- "-$orchard_vision_pid" 2>/dev/null || true
  for attempt in {1..20}; do
    if ! kill -0 -- "-$orchard_vision_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 -- "-$orchard_vision_pid" 2>/dev/null; then
    kill -TERM -- "-$orchard_vision_pid" 2>/dev/null || true
  fi
  wait "$orchard_vision_pid" 2>/dev/null || true
  orchard_vision_pid=""
}
trap orchard_cleanup HUP INT TERM EXIT

setsid ros2 run apple_vision apple_detector "${orchard_ros_args[@]}" &
orchard_vision_pid=$!
set +e
wait "$orchard_vision_pid"
orchard_status=$?
set -e
orchard_cleanup
exit "$orchard_status"

