#!/usr/bin/env bash
set -eo pipefail

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source /home/nvidia/orchard_experiment/setup_env.sh
cd "$RL_ORCHARD_ROOT"

colcon --log-base "$RL_ORCHARD_ROOT/log" build \
  --base-paths "$RL_ORCHARD_ROOT/src" \
  --build-base "$RL_ORCHARD_ROOT/build" \
  --install-base "$RL_ORCHARD_ROOT/install" \
  --packages-select orchard_picker apple_vision
