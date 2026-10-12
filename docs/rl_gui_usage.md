# RL 数据采集 GUI 使用说明

本文说明如何使用复制版 GUI 进行苹果抓取数据采集。

原 GUI 位于：

```text
/home/nvidia/orchard_experiment
```

RL 采集版 GUI 位于：

```text
/home/nvidia/orchard_experiment_rl_collect
```

原 GUI 不要修改。RL GUI 继续使用原有相机显示、机械臂状态显示和专家抓取流程，
Recorder 只负责旁路记录数据。

## 1. 启动 GUI

先确认机械臂网络已经由人工配置完成，再执行：

```bash
cd /home/nvidia/orchard_experiment_rl_collect
./run_gui.sh
```

GUI 启动不会自动让机械臂运动，也不会自动调用 `run_once`、Home、MoveJ、MoveL 或夹爪。

## 2. 连接相机和机械臂

按原 GUI 的操作方式依次点击：

1. **连接相机**
2. **连接机械臂**

确认界面能够看到：

- 苹果实时图像和检测框
- Apple Camera XYZ
- Apple Base XYZ
- Robot Connected
- Robot Error Code
- J1～J6
- TCP pose
- Pick Task 状态

RL Recorder 保存的是视觉节点提供的原始 RGB 和 aligned `uint16` depth，GUI 上显示的
annotated image 只用于人工观察，不会写入训练图像。

## 3. 启动 Recorder

点击 **开始记录**。

GUI 会启动或连接已有的 `rl_data_recorder`，成功后按钮变为 **停止记录**，状态显示为
“记录中”。Recorder 可以在整个实验期间持续运行，不需要每条 episode 重新启动。

如果 Recorder 已经由独立终端启动，GUI 只使用它，不会重复打开第二个 Recorder。

## 4. 运行 Preflight

点击 **运行 Preflight**，等待终端输出 `Overall PASS`。

当前只读检查包括：

- RGB 和 depth 是否新鲜
- Apple camera/base target 是否有效
- `/rm_json_bridge/connected == true`
- RobotState、TCP、joint 是否新鲜
- joint 数量是否为 6
- joint source callback rate
- `/rl_data_recorder` 是否存在
- 磁盘空间

Preflight 采用当前数据配置：

```text
record_hz = 4 Hz
视觉 freshness = 150 ms
机器人/TCP/joint freshness = 300 ms
```

如果相机、机械臂或 Recorder 状态发生变化，旧的 Preflight PASS 会失效，需要重新运行。

## 5. 采集一条专家 episode

先人工放置苹果，确认苹果在画面中、目标坐标合理、Robot Error Code 为 0。

然后点击原有的 **开始抓取** 按钮。

当记录开关处于“记录中”时，调用顺序为：

```text
Recorder prepare_episode
        ↓
Recorder 进入 ARMED/RECORDING
        ↓
原有 /pick_task_manager/run_once
        ↓
传统专家抓取流程
        ↓
Recorder finalize
```

现有 MoveJ、MoveJ_P、MoveL、Home、Gripper 和状态机逻辑不会被 Recorder 重写。

正常结束后，Recorder 保存独立 HDF5 episode。失败、急停或异常结束的 episode 也会保留，
但会标记为 `episode_valid=false`，默认训练筛选会跳过。

## 6. 停止记录

没有活动 episode 时，点击 **停止记录** 只关闭记录开关，之后点击抓取不会创建新的 RL
episode。

如果当前正在记录，GUI 会先请求 Recorder 将当前 episode 标记为 INVALID，然后关闭记录
开关。这个操作不会：

- 停止相机
- 停止机械臂 ROS 节点
- Home
- MoveJ 或 MoveL
- 操作夹爪
- 调用急停

## 7. 发生危险时急停

遇到机械臂异常时，立即点击红色 **机械臂急停**，必要时同时使用实体急停。

GUI 调用链为：

```text
GUI
 → /rm_json_bridge/emergency_stop
 → rm_json_bridge
 → set_arm_emergency_stop
 → RealMan controller
```

只有收到以下响应才显示急停确认：

```json
{
  "command": "set_arm_emergency_stop",
  "set_state": true
}
```

急停不会自动解除。急停发生时，当前 episode 会保留但标记为 INVALID，不进入默认训练集。

## 8. 检查最新 episode

查找最新文件：

```bash
ls -1t /home/nvidia/RL-Orchard/rl_dataset/episode_*.hdf5 | head -1
```

检查数据：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/inspect_rl_dataset.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5
```

回放数据：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/replay_episode.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5
```

Replay 只读取 HDF5，不初始化 ROS，也不会控制机械臂。

## 9. 结束实验

实验结束后：

1. 确认没有活动抓取任务。
2. 点击 **停止记录**，或在 Recorder 终端按 `Ctrl+C`。
3. 确认最新 episode 已完成重命名为 `.hdf5`，而不是仍停留在 `.tmp.hdf5`。
4. 使用 inspector 检查 `Quality pass`、`Episode valid`、joint delta 和 goal 字段。

默认训练数据筛选会排除：

- `episode_valid=false`
- `discarded=true`
- QC 失败
- 样本数过少
- 持续时间过短
- terminal state 为 `IDLE` 或未知

