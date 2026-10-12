#!/usr/bin/env bash
set -eo pipefail

if [[ "${ORCHARD_ARM_ENABLE:-}" != "YES" ]]; then
  echo "未启动机械臂。确认急停、工作区、Home 位姿和夹爪安全后执行：" >&2
  echo "ORCHARD_ARM_ENABLE=YES /home/nvidia/RL-Orchard/scripts/run_instrumented_arm_stack.sh" >&2
  exit 2
fi

RL_ORCHARD_ROOT="/home/nvidia/RL-Orchard"
source "$RL_ORCHARD_ROOT/scripts/setup_rl_env.sh"
set -u

ORCHARD_ARM_IP="${ARM_IP:-192.168.1.18}"
ORCHARD_ARM_PORT="${ARM_PORT:-8080}"
ping -c 1 -W 1 "$ORCHARD_ARM_IP" >/dev/null
nc -z -w 2 "$ORCHARD_ARM_IP" "$ORCHARD_ARM_PORT"

echo "机械臂网络已通过。使用 RL-Orchard overlay；控制参数与原配置副本一致。"
echo "本脚本不会调用 home、run_once 或 start。"
exec ros2 launch orchard_picker orchard_picker.launch.py \
  config:="$RL_ORCHARD_ROOT/src/orchard_picker/config/picker_params.yaml"

