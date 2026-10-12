#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "请使用: source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh" >&2
  exit 2
fi

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source /home/nvidia/orchard_experiment/setup_env.sh

if [[ ! -f "$RL_ORCHARD_ROOT/install/setup.bash" ]]; then
  echo "RL overlay 尚未构建: $RL_ORCHARD_ROOT/install/setup.bash" >&2
  return 1
fi

# The V2 single-command launcher uses the authoritative ros2_ws orchard_picker
# package. Its generated services and launch files must not be shadowed by the
# RL overlay, whose orchard_picker package is a different collection overlay.
if [[ "${RL_ORCHARD_SKIP_RL_OVERLAY:-0}" != "1" ]]; then
  source "$RL_ORCHARD_ROOT/install/setup.bash"
fi
export PYTHONPATH="$RL_ORCHARD_ROOT/vendor/python${PYTHONPATH:+:$PYTHONPATH}"
export ROS_LOG_DIR="$RL_ORCHARD_ROOT/log/ros"
mkdir -p "$ROS_LOG_DIR"
