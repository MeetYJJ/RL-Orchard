#!/usr/bin/env bash
set -eo pipefail

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source /home/nvidia/orchard_experiment/setup_env.sh

python -m pip install \
  --no-deps \
  --upgrade \
  --target "$RL_ORCHARD_ROOT/vendor/python" \
  -r "$RL_ORCHARD_ROOT/requirements-rl.txt"

