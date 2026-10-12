#!/usr/bin/env bash
set -eo pipefail

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source "$RL_ORCHARD_ROOT/scripts/setup_rl_env.sh"

CONFIG="${RL_COLLECTION_CONFIG:-$RL_ORCHARD_ROOT/src/orchard_picker/config/rl_collection.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Collection 配置不存在: $CONFIG" >&2
  exit 2
fi

if [[ "${ORCHARD_ARM_ENABLE:-}" != "YES" || "${ORCHARD_COLLECTION_ENABLE:-}" != "YES" ]]; then
  echo "Collection 处于 DRY RUN：未同时设置两个 YES 门控，不会调用任何运动/抓取 service。"
  exec ros2 run orchard_picker rl_collection_orchestrator.py \
    --config "$CONFIG" --dry-run "$@"
fi

exec ros2 run orchard_picker rl_collection_orchestrator.py \
  --config "$CONFIG" "$@"
