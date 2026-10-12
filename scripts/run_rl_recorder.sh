#!/usr/bin/env bash
set -eo pipefail

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source "$RL_ORCHARD_ROOT/scripts/setup_rl_env.sh"

CONFIG="${RL_RECORDER_CONFIG:-$RL_ORCHARD_ROOT/src/orchard_picker/config/rl_data_recorder.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Recorder 配置不存在: $CONFIG" >&2
  exit 2
fi

# This process only subscribes to data and, when necessary, makes the existing
# read-only get_arm_state request.  It exposes no robot motion service/client.
exec ros2 run orchard_picker rl_data_recorder.py \
  --ros-args --params-file "$CONFIG"
