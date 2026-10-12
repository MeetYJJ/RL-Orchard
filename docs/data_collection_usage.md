# RL-Orchard 数据采集使用说明

本工具采集现有 Traditional Controller 的 Expert demonstration。Recorder 只读观测和写盘，不使用 SAC 控制机械臂，也不会发送运动、夹爪、Home、start 或 run_once 命令。

## 1. 数据内容

每个 episode 保存为一个 HDF5 文件：

```text
/home/nvidia/RL-Orchard/rl_dataset/episode_XXXXXX.hdf5
```

主要数据包括：

- RGB、aligned uint16 depth、相机 metadata
- J1～J6（radian）、TCP pose、robot error、gripper measured state
- task phase、active、apple camera/base target、confidence、bbox
- Expert command event 和 command source
- `actions/executed_delta_tcp`
- `actions/executed_delta_joint[t] = q[t+1] - q[t]`
- `teacher/pregrasp_pose_base`、`teacher/grasp_pose_base`、
  `teacher/approach_vector_base`、`teacher/pregrasp_offset_m`
- `task/reached_pregrasp`
- `reward_terms/distance_to_pregrasp_m`、`reward_terms/progress_m`、
  `reward_terms/action_norm`

`executed_delta_joint` 的最后一行没有 next state，因此对应的
`valid/executed_delta_joint` 为 `false`。Cartesian MoveJ_P/MoveL 没有真实 joint
target 时，`valid/commanded_joint_target` 保持 `false`，不会执行 IK 猜测。

当前默认 joint source：

```text
joint_state_source = rm_json_bridge
topic = /rm_json_bridge/arm_state
joint_pos_rad = rad
```

`joint_state_observed_rate_hz` 使用 Recorder 收到 joint-state source callback 的
monotonic timestamp 跨度计算，不使用 4/10 Hz Recorder tick，也不把重复缓存帧当成
新的 Recorder sample；它表示 source callback cadence，不等同于控制器内部伺服频率。
同一文件还保存 `time/joint_state_timestamp_ns` 和 `time/joint_state_age_ms`。

从 `apple_rl_v1.2` 起，goal 由真实 ExpertAction 的 `MOVE_PRE_GRASP` 与 `MOVE_GRASP`
事件冻结。目标 pose 使用 `base_link` 下的 `robot_end`，approach vector 是从
pre-grasp 指向 grasp 的单位向量，offset 是实际位置距离。缺少完整事件时保存 NaN，
不运行 IK。`rewards` 仍为 0 占位；`reward_terms` 保存离线计算距离进展和动作范数的
原始量。

Recorder 采样频率由 `rl_data_recorder.yaml` 的 `record_hz` 控制（当前配置为 4 Hz；
需要 10 Hz 时只调整数据层配置并重新启动 Recorder）。HDF5 文件会先写成
`.tmp.hdf5`，episode 正常结束后再原子重命名；异常退出时保留临时文件。

## 2. 构建 overlay

```bash
cd /home/nvidia/RL-Orchard
./scripts/build_overlay.sh
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
```

不要在本流程中修改或重新构建 `/home/nvidia/orchard_experiment`、
`/home/nvidia/ros2_ws` 或 `/home/nvidia/yoloros2_ws`。

## 2.1 RL 数据采集 GUI（推荐的人工真机流程）

RL 采集版 GUI 是 `/home/nvidia/orchard_experiment_rl_collect` 的独立副本，
不会修改原 GUI。它只启动/观察 RL overlay，不重写现有 Pick Task Manager 的抓取
轨迹；GUI 预览使用 `/apple/annotated_image`，Recorder 仍从 `/apple_vision/rgb`
和 `/apple_vision/aligned_depth` 保存未画框的原始 RGB 与 aligned `uint16` depth。

```bash
cd /home/nvidia/orchard_experiment_rl_collect
./run_gui.sh
```

复制版尽量保持原 GUI 的连接、预览和抓取操作，不承担复杂 ROS supervisor。打开 GUI
不会自动启动相机或机械臂；相机和机械臂仍由原来的按钮控制。RL 额外只有一个记录开关。

按以下顺序点击：

1. 按原 GUI 的 **连接相机**、**连接机械臂**，确认预览、Camera/Base XYZ、J1～J6、TCP
   和任务状态正常。相机按钮启动的是完整 RL Vision（RealSense + YOLO）节点，不是
   第二个相机；GUI 自身只订阅画面。
2. 点击复制版新增的 **开始记录**。GUI 确认 `/rl_data_recorder/set_enabled` 成功后，
   状态变为“已启用”。Recorder 进程只启动一次并可长期运行。
3. 人工摆放苹果，点击原来的 **开始抓取**。记录开启时，GUI 先调用
   `/rl_data_recorder/prepare_episode`，得到成功响应后才调用原来的
   `/pick_task_manager/run_once`；记录关闭时完全执行原抓取流程。
4. 正常结束后 Recorder 自动 finalize 为 VALID；失败、急停或点击 **停止记录** 时，
   当前 episode finalize 为 INVALID，但 HDF5 保留。

红色 **急停**按钮走唯一的 `rm_json_bridge` ROS service
`/rm_json_bridge/emergency_stop`，由 bridge 通过已有 TCP/JSON client 发送
`{"command":"set_arm_emergency_stop","emergency_stop":true}`，并等待控制器返回
`command="set_arm_emergency_stop"` 且 `set_state=true` 后才报告“已确认”。这是软件
急停状态，不是实体急停的替代品；本 GUI 不提供自动解除急停。急停期间的 episode 会保留
并写入 `episode_valid=false`、`discarded=true`、`termination_type=emergency_stop`，
默认数据统计和训练筛选会排除它。

**停止记录**只关闭记录开关；如果当前有 episode，会先调用
`/rl_data_recorder/abort_episode` 标记 INVALID，然后停止创建新的 episode。它不会停止
相机、机械臂、Home、MoveJ、MoveL 或夹爪。

## 3. 启动三个基础进程

每条命令使用独立终端。

终端 A：启动视觉 overlay。该节点独占 D435i pipeline。

```bash
CAMERA_SERIAL=908212070350 \
/home/nvidia/RL-Orchard/scripts/run_rl_vision.sh
```

终端 B：启动已有机械臂控制栈。脚本不会自动 Home、start 或 run_once。

```bash
ORCHARD_ARM_ENABLE=YES \
/home/nvidia/RL-Orchard/scripts/run_instrumented_arm_stack.sh
```

终端 C：启动只读 Recorder。

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_recorder.sh
```

默认配置位于：

```text
src/orchard_picker/config/rl_data_recorder.yaml
```

## 4. 只读预检和静止测试

先让机械臂保持静止。苹果需要已经人工放置并被视觉节点检测到，否则 target 检查会失败。

终端 D 执行：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh

python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/check_real_data_pipeline.py \
  --joint-source rm_json_bridge \
  --record-hz 4 \
  --duration 10 \
  --max-sensor-age-ms 150 \
  --max-robot-age-ms 300
```

预检只订阅 ROS topic，不创建机器人控制 client。它检查：

- robot connected 和 robot error code
- RGB、aligned depth、camera metadata
- RobotState/TCP 和 joint shape `[6]`
- joint freshness、实际时间戳更新率和 Recorder 是否在线
- apple camera/base target
- 数据盘剩余空间

预检必须全部 `PASS`。它还会输出静止期间 joint delta 的 mean、std 和 max abs；不要把微小反馈噪声强行过滤为零。

如果 joint rate 小于 10 Hz，不要采集。当前自定义 bridge 不是独立高频反馈驱动，必要时先确认官方 `/joint_states` 是否存在；只有确认六个 joint name 后，才能把配置切换为 `sensor_msgs_joint_state`。

## 5. 采集一次 Expert episode

预检通过后，使用当前已经验证的 GUI Expert 流程执行一次抓取。GUI 不应再启动第二套相机或机械臂进程。

episode 生命周期由真实 task state 驱动。Recorder 不会把 IDLE 或手动移动开成
episode：`active` 上升后还必须等到 `OPEN_GRIPPER` 或 `MOVE_PRE_GRASP` 才真正创建
文件；如果 active 只是短暂变化而没有进入这些 phase，则不会产生 episode。

正常 episode 的状态转换为：

```text
/pick_task_manager/active: false -> true  等待真实执行 phase
/pick_task_manager/state: OPEN_GRIPPER 或 MOVE_PRE_GRASP  开始
/pick_task_manager/active: true -> false  等待 outcome 后结束
```

episode 开始帧会自动写入当时新鲜的 `initial_joint_pos`、`initial_tcp_pose` 和
initial apple camera/base xyz、id、confidence、bbox。每个文件写入同一份全局
`task state_mapping`，不会因本 episode 只出现少数 phase 而重新编号。

默认 `min_episode_samples=5`、`min_episode_duration_sec=0.5`。样本数过少、持续时间
过短或 terminal state 为 `IDLE`/`UNKNOWN` 时，文件仍保留但 QC invalid，且
`task_success/success` 强制为 `false`。

采集中不要启动 SAC，也不要调用 `movej_follow` 或 `movej_canfd`。唯一运动来源是现有 Traditional Expert Controller。

Recorder 日志中应出现：

```text
[RL Recorder] episode_XXXXXX started
...
[RL Recorder] episode_XXXXXX finalized
```

## 6. 检查和人工物理标签

查找最新完整 episode：

```bash
EPISODE=$(
  find /home/nvidia/RL-Orchard/rl_dataset \
    -maxdepth 1 -name 'episode_*.hdf5' ! -name '*.tmp.hdf5' \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
)
```

运行 QC/Inspector：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/inspect_rl_dataset.py \
  "$EPISODE"
```

重点确认：

- RGB、depth、joint、TCP valid ratio
- joint source、joint age、joint update rate
- `executed_delta_joint` valid ratio
- `policy_segment_rgb_valid_ratio`
- `policy_segment_depth_valid_ratio`
- `policy_segment_all_valid_ratio`
- `q[t] + executed_delta_joint[t] == q[t+1]`
- 最后一行 action invalid、最后一行 done=true
- phase sequence、Expert event count
- QC 为 `True`

填写真实物理抓取结果：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/label_episode.py \
  "$EPISODE"
```

交互输入：

```text
1  确实抓住苹果
0  抓取失败
u  无法判断
```

随后填写 failure reason 和 operator note。失败、中止、安全停止 episode 不删除。

## 7. 回放和覆盖率统计

图像回放：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/replay_episode.py \
  "$EPISODE"
```

快捷键：`Space` 暂停，左右方向键单帧，`Q` 或 `Esc` 退出。回放不会初始化 ROS，也不会控制机器人。

无图形环境检查：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/replay_episode.py \
  "$EPISODE" --headless
```

统计数据覆盖度：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/report_collection_coverage.py \
  /home/nvidia/RL-Orchard/rl_dataset
```

## 8. 可选的只读版本查询

控制器在线且确认使用正确版本命令后，可以查询版本。工具只接受白名单中的只读命令：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/query_realman_version.py \
  get_arm_software_info

python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/query_realman_version.py \
  get_joint_software_version
```

查询失败或返回内容不匹配时，不要改用任意 raw JSON。版本查询不是 Recorder 启动的必要步骤。

## 9. 常见失败处理

`RGB/depth stale`：确认视觉进程和 topic 名称，没有启动第二个 RealSense pipeline。

`joint rate FAIL`：确认 `/rm_json_bridge/arm_state` 的实际更新频率；不要把 5 Hz 数据当成 10 Hz 数据。检查官方 `/joint_states` 前先核对关节名称。

`target invalid`：重新放置苹果，等待相机检测和 `target_base` 发布，再重新运行预检。

出现 `.tmp.hdf5`：说明进程中断或 episode 没有正常 finalize。保留文件，不要覆盖；先检查 Recorder 日志，再决定是否恢复或重新采集。

`quality_pass=false`：文件保留。先阅读 `quality_reason`，修复数据源问题后重新采集，不要手工把 QC 改成通过。
