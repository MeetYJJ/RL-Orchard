# RML63-B 苹果抓取只读数据采集 Overlay

本目录是一套独立 ROS 2 Foxy overlay。它不修改
`/home/nvidia/orchard_experiment`、`/home/nvidia/ros2_ws`、
`/home/nvidia/yoloros2_ws` 或现有 Conda 环境。只有显式 source
`scripts/setup_rl_env.sh` 并运行本目录脚本时，采集版节点才会生效。

## 1. 源码与 ROS 接口审计结果

审计日期：2026-09-18。审计以实际源码为准；执行运行时 ROS 图检查时，
系统只有 `/rosout` 和 `/parameter_events`，没有业务节点在线，因此下表的
类型来自源码创建的 publisher/service，首次联机时仍需用 `topic info -v`
复核。

现有可直接取得的数据：

| 数据 | 原接口 | 实际类型/来源 |
|---|---|---|
| 苹果三维列表 | `/apple_positions` | `apple_interfaces/msg/ApplePositionArray`，相机坐标 |
| 抓取目标 | `/apple/target_point` | `geometry_msgs/msg/PointStamped` |
| GUI 图像 | `/apple/annotated_image` | `sensor_msgs/msg/CompressedImage`，已画框 JPEG，不能用于训练主图像 |
| 目标教师信息 | `/apple/target_info` | `std_msgs/msg/String` JSON，原有 id/confidence/camera xyz |
| Base 目标 | `/pick_task_manager/target_base` | `geometry_msgs/msg/PointStamped`，由已验证的 TF 变换结果发布 |
| 抓取 phase | `/pick_task_manager/state` | `std_msgs/msg/String`，transient-local |
| 抓取 active | `/pick_task_manager/active` | `std_msgs/msg/Bool`，transient-local |
| 机器人连接 | `/rm_json_bridge/connected` | `std_msgs/msg/Bool`，transient-local |
| 原始收发 JSON | `/rm_json_bridge/raw_rx`、`raw_tx` | `std_msgs/msg/String` |
| J1～J6、控制器 TCP pose、错误码 | `/rm_json_bridge/get_arm_state` | `orchard_picker/srv/GetArmState`；joint 原始单位 degree，pose 为 `base_link` 中 controller `robot_end` |
| 记录用 RobotState | `/rm_json_bridge/arm_state` | `orchard_picker/msg/RobotState`；`joint_pos_rad[6]` 已是 rad，发布由已有只读状态查询触发 |
| TCP TF | `/tf` | `tcp_tf_publisher` 以 5 Hz 调用上述只读 service，发布 `base_link -> robot_end` |
| 运动/夹爪专家命令 | `/rm_json_bridge/raw_tx` | 原始 JSON 可见，但原系统没有结构化 action topic |

原始控制栈缺失的部分由本 overlay 补充为只读 topic：未画框 RGB、aligned Z16
depth、相机 depth scale metadata、结构化 RobotState、ExpertAction 和 TaskOutcome。
EG2 经过真机验证的实测位置/力/电流/运动状态仍可能不可用，因此保持 unknown/NaN。

真实状态机 phase 为：`IDLE`、`DRY_RUN_DONE`、`OPEN_GRIPPER`、
`MOVE_PRE_GRASP`、`MOVE_GRASP`、`CLOSE_GRIPPER`、`GRASPED_HOLDING`、
`TWIST_J6`、`RETREAT`、`HOME`、`RELEASE_GRIPPER`、`DONE`、`FAILED`、
`ABORT_REQUESTED`、`SAFETY_STOP`。抓取流程的状态机成功终态是 `DONE`、
`GRASPED_HOLDING` 或 `DRY_RUN_DONE`；`physical_grasp_success` 当前始终为
`-1 (unknown)`，不会把状态机完成伪装成物理抓取成功。

实际 service 还包括：

- `/rm_json_bridge/movej` (`MoveJ`)
- `/rm_json_bridge/movej_p`、`/rm_json_bridge/movel` (`MovePose`)
- `/rm_json_bridge/gripper_release`、`gripper_pick`、`gripper_position`
- `/rm_json_bridge/stop`、`raw_json`、`set_tool_voltage`
- `/pick_task_manager/run_once`、`start`、`abort`、`home`

Recorder 没有创建或调用任何 motion、gripper、Home、start、run_once service。
唯一 service client 是异步、限频的只读 `get_arm_state` fallback。

## 2. 最小改动架构

```text
本地 apple_vision 副本
  同一 rs.pipeline / 同一次 align
  ├─ camera acquisition thread
  │    └─ bounded latest-frame cache
  ├─ independent raw publisher
  │    ├─ /apple_vision/rgb          bgr8 raw
  │    └─ /apple_vision/aligned_depth 16UC1
  ├─ YOLO/ByteTrack detector timer (may skip stale frames)
  └─ /apple_vision/camera_metadata  depth scale/serial

本地 orchard_picker 副本
  ├─ rm_json_bridge：复用既有 get_arm_state 返回值发布 RobotState
  └─ pick_task_manager：命令下发前旁路发布 ExpertAction，结束前发布 TaskOutcome

上述 topic + 原 state/active/target topics
                    │
                    ▼
              rl_data_recorder
     callback 只更新 latest cache / event cache
                    │ 10 Hz
                    ▼
             bounded sample queue
                    │
                    ▼
            独立 HDF5 writer thread
                    │
       episode_NNNNNN.tmp.hdf5
                    │ 正常 finalize + os.replace
                    ▼
           episode_NNNNNN.hdf5
```

遥测 publish 全部有异常隔离；publisher 失败只记 warning，不会阻止原运动
service 调用、状态 service 返回或现有 GUI 图像发布。

视觉 raw stream 的发布线程使用 ROS 2 Foxy `Image.data` 所需的
`array('B')` 连续缓冲，避免把 640×480 图像按 Python 字节逐个转换。配置中的
`training_publish_hz: 30.0` 是 raw stream 上限；latest cache 会丢弃旧帧，不会
积压内存。YOLO/ByteTrack 只消费最新缓存帧，检测较慢时会跳帧但不再降低 raw
RGB-D 的采集/发布链。

只读验证（需要先启动 `scripts/run_rl_vision.sh`）可使用较小窗口并在终端模式
观察启动后的稳态值：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
ros2 topic hz --window 20 --wall-time /apple_vision/rgb
ros2 topic hz --window 20 --wall-time /apple_vision/aligned_depth
ros2 topic delay /apple_vision/rgb
ros2 topic delay /apple_vision/aligned_depth
```

启动初期会包含 DDS discovery 和订阅端预热间隔，应以随后连续窗口为准。

## 3. 新增文件

- `src/orchard_picker/msg/{RobotState,ExpertAction,TaskOutcome}.msg`
- `src/orchard_picker/scripts/rl_data_recorder.py`
- `src/orchard_picker/src/orchard_picker/rl_data/episode_writer.py`
- `src/orchard_picker/src/orchard_picker/rl_data/action_utils.py`
- `src/orchard_picker/config/rl_data_recorder.yaml`
- `src/orchard_picker/scripts/inspect_rl_dataset.py`
- `src/orchard_picker/scripts/replay_episode.py`
- `scripts/setup_rl_env.sh`、`build_overlay.sh`、`install_local_deps.sh`
- `scripts/run_rl_vision.sh`、`run_instrumented_arm_stack.sh`、`run_rl_recorder.sh`
- `tests/create_mock_episode.py`、`tests/test_action_utils.py`
- `src/orchard_picker/scripts/rl_collection_orchestrator.py`
- `src/orchard_picker/config/rl_collection.yaml`
- `src/orchard_picker/src/orchard_picker/rl_data/{schema,dataset_tools,collection_planning}.py`
- `src/orchard_picker/src/orchard_picker/rl_data/joint_state.py`
- `src/orchard_picker/scripts/{label_episode,report_collection_coverage}.py`
- `src/orchard_picker/scripts/{check_real_data_pipeline,query_realman_version}.py`
- [简明数据采集使用说明](docs/data_collection_usage.md)
- `scripts/run_rl_collection.sh`
- `tests/{mock_data,test_collection_planning,test_dataset_collection}.py`

本地副本中最小修改的原文件：

- `apple_detector_node.py`：单一 `rs.pipeline()` 由采集线程读取并对齐，latest
  frame cache 与独立 raw publisher 解耦 `/apple_vision/rgb`、`aligned_depth`
  和 YOLO/ByteTrack；RGB/depth 仍来自同一 frameset 并共享采集 timestamp。
  检测、目标和 GUI 逻辑保持不变，不创建第二个 `rs.pipeline()`。
- `apple_detector.yaml`：增加三个训练数据 topic 参数。
- `rm_json_bridge.py`：把已发生的状态查询结果转换为 radian 后旁路发布；不增加
  TCP connection。
- `pick_task_manager.py`：增加只读 expert action/task outcome publisher；不改命令
  数值、顺序、轨迹、安全检查或状态机。
- `CMakeLists.txt`、`package.xml`：注册消息、脚本和依赖。
- `rl_data_recorder.py`、`episode_writer.py`：增加一次性 collection metadata、
  recording gate、control-mode/policy mask、joint transition/command source、QC/manifest；
  没有增加任何 motion client。
- `inspect_rl_dataset.py`、`replay_episode.py`：显示新标签/mask，并兼容旧 episode。

## 4. 构建与纯离线测试

```bash
cd /home/nvidia/RL-Orchard
./scripts/install_local_deps.sh   # 只写 vendor/python，不改 Conda
./scripts/build_overlay.sh        # build/install/log 均在 RL-Orchard

source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python -m unittest discover -s /home/nvidia/RL-Orchard/tests -p 'test_*.py' -v
python /home/nvidia/RL-Orchard/tests/create_mock_episode.py \
  --output-root /tmp/apple_rl_mock
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/inspect_rl_dataset.py \
  /tmp/apple_rl_mock/episode_000001.hdf5
```

已完成的静态测试：Python 3.8 `py_compile`、两个 package 的 `colcon build`、
ROS 自定义消息导入/`ros2 interface show`、四元数相对旋转单测、12 帧 mock
episode、inspector。没有启动 ROS 业务节点、相机或任何真机 motion service。

## 5. 第一次只读联机验证（不运动机械臂）

Terminal A，启动本地视觉副本。不要同时点击 GUI 的“连接相机”，避免第二个
进程打开 D435i：

```bash
CAMERA_SERIAL=908212070350 \
  /home/nvidia/RL-Orchard/scripts/run_rl_vision.sh
```

Terminal B，仅在完成现有急停/工作区检查后连接机械臂。脚本本身不会调用
Home、start 或 run_once：

```bash
ORCHARD_ARM_ENABLE=YES \
  /home/nvidia/RL-Orchard/scripts/run_instrumented_arm_stack.sh
```

Terminal C：

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_recorder.sh
```

Terminal D，只读确认：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
ros2 node list
ros2 topic list -t
ros2 topic info /apple_vision/rgb -v
ros2 topic info /apple_vision/aligned_depth -v
ros2 topic info /rm_json_bridge/arm_state -v
ros2 topic info /pick_task_manager/state -v
ros2 topic info /pick_task_manager/active -v
ros2 topic echo /apple_vision/camera_metadata --once
ros2 topic echo /rm_json_bridge/arm_state --once
```

预期 Recorder 依次报告 RGB、aligned depth、robot state、task state、active、
teacher info received。此阶段不要调用任何运动 service。

GUI 仍可按原命令启动并订阅同名 topic/service：

```bash
cd /home/nvidia/orchard_experiment
source /home/nvidia/orchard_experiment/setup_env.sh
./run_gui.sh
```

Recorder 启动后，先运行只读 joint/data preflight（默认观察 5 秒）：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/check_real_data_pipeline.py \
  --joint-source rm_json_bridge --duration 5
```

它只订阅 RGB、aligned depth、camera metadata、`RobotState`、target 和连接状态，
检查 joint shape/freshness/effective rate、TCP、目标和磁盘；任一 FAIL 都不要开始
采集。静止机械臂保持 5--10 秒，报告 joint delta 的 mean/std/max-abs，不把微小
反馈噪声硬过滤为零。

本地视觉/机械臂节点已经运行时，不要再通过 GUI 启动第二套相机或机械臂进程；
抓取、Home、急停 service 本身仍由 GUI 按原方式调用本地 overlay 中同一份控制
逻辑。

## 6. 第一次真实抓取采集

1. 保持 A/B/C 三个进程正常，先通过上面的 preflight 和静止测试。
2. 由人工按当前已验证的 GUI Expert 流程选择目标并执行一次抓取；不要从测试脚本调用
   motion service。
   本轮不启用 SAC、`movej_follow` 或 `movej_canfd`；唯一运动来源仍是现有 Expert。
3. `active false -> true` 后，Recorder 先等待 `OPEN_GRIPPER` 或 `MOVE_PRE_GRASP`；
   只有进入真实执行 phase 才创建 episode。`true -> false` 且 outcome 已发布后
   finalize。异常退出保留 `.tmp.hdf5`。
4. 检查最新文件：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
EPISODE=$(ls -1t /home/nvidia/RL-Orchard/rl_dataset/episode_*.hdf5 | head -1)
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/inspect_rl_dataset.py "$EPISODE"
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/replay_episode.py "$EPISODE"
```

Replay 键位：Space 暂停，左右方向键单帧，Q/Esc 退出。它完全不初始化 ROS，
不会控制机器人。

## 7. HDF5 `apple_rl_v1` / `apple_rl_v1.1` / `apple_rl_v1.2` 结构

```text
obs/{rgb,depth,joint_pos,tcp_pose,gripper_state,
     gripper_position,gripper_force,gripper_current,gripper_moving,
     robot_error_code}
next_obs/{joint_pos,tcp_pose}
actions/{executed_delta_tcp,executed_delta_joint,commanded_joint_target,
         command_source,gripper}
teacher/{apple_camera_xyz,apple_base_xyz,apple_id,confidence,bbox,target_valid,
         grasp_pose_base,pregrasp_pose_base,approach_vector_base,pregrasp_offset_m,
         action_type,action_phase,target_tcp,target_joint,gripper_command,
         command_speed,command_force,action_event_count}
task/{state,active,reached_pregrasp}
time/{timestamp_ns,rgb_timestamp_ns,depth_timestamp_ns,
      robot_state_timestamp_ns,joint_state_timestamp_ns,joint_state_age_ms}
valid/{rgb,depth,robot_state,apple_target,action,expert_action,
       gripper_measured,joint_state,executed_delta_joint,
       commanded_joint_target,expert_joint_action,policy_action}
episode_meta/{initial_joint_pos,initial_tcp_pose,initialization_offset,
              initial_apple_camera_xyz,initial_apple_base_xyz,
              initial_apple_id,initial_apple_confidence,initial_apple_bbox}
rewards
dones
reward_terms/{distance_to_pregrasp_m,progress_m,action_norm}
debug/raw_tx                 # 仅 save_raw_command_debug=true 时
```

`obs/rgb` 为 RGB uint8；`obs/depth` 为 aligned uint16；关节为 radian；TCP 为
`[x,y,z,qx,qy,qz,qw]`。`executed_delta_tcp[t]` 从 `tcp[t]` 与 `tcp[t+1]`
计算，相对旋转使用 `inverse(q_t) * q_{t+1}` 转 rotation vector；同一采样区间内
的 expert event 回填到第 `t` 行。最后一行 `valid/action=false`，仅最后一行
`dones=true`。reward 当前全 0，可离线重算。

从 `apple_rl_v1.1` 起，`executed_delta_joint[t]` 严格定义为
`obs/joint_pos[t+1] - obs/joint_pos[t]`（单位 rad），代表 Expert 在该采样区间
真实发生的 joint displacement；最后一行无 next state，因此 valid=false、数值为
NaN。它是 BC/offline-RL 候选标签，不是当前 SAC 在线 requested action。
`commanded_joint_target` 只有真实结构化 MoveJ/`movej_follow`/`movej_canfd` 事件
携带六个 joint target 时才 valid；MoveJ_P/MoveL 的 Cartesian target 不做 IK，保持
NaN/false。`command_source` 使用固定字符串枚举。旧 episode 没有这些 dataset 时仍
可由 inspector/replay 打开。

枚举为 `NONE`、`MOVEJ`、`MOVEJ_P`、`MOVEL`、`MOVEJ_FOLLOW`、`MOVEJ_CANFD`、
`GRIPPER`、`OTHER`；记录器只根据收到的结构化 `ExpertAction` 填写，不从 raw JSON
或 Cartesian target 推断 joint target。

从 `apple_rl_v1.2` 起，Recorder 从真实结构化 ExpertAction 的
`MOVE_PRE_GRASP` 与 `MOVE_GRASP` 事件冻结 SAC goal。两个 `target_tcp` 都是
`base_link` 下实际 `robot_end` pose，写入 `episode_meta/` 和每帧对齐的 `teacher/`
字段。`approach_vector_base` 是从 pre-grasp 指向 grasp 的单位向量，
`pregrasp_offset_m` 是两者实际位置距离，不假定为 0.08 m。goal 在一个 episode 内
保持不变；缺少完整事件时字段保持 NaN，不运行 IK，也不猜测目标。

`task/reached_pregrasp` 与 root attribute `reached_pregrasp` 在观察到专家进入
`MOVE_GRASP` 前置阶段时置 true。第一版低维 SAC 状态可由每条 transition 恢复为：

```text
[obs/joint_pos(6), obs/tcp_pose[:3](3), pregrasp_pose_base[:3] - obs/tcp_pose[:3](3)]
```

动作仍是 `actions/executed_delta_joint`。`reward_terms/*` 保存离线重算所需的距离、
距离改善和 joint action 范数；`rewards` 仍是占位 0，reward 权重留给训练阶段决定。

Recorder 当前选择 `joint_state_source=rm_json_bridge` 时使用
`RobotState.joint_pos_rad`；因此新 episode 的 `source_joint_unit` 和
`stored_joint_unit` 都是 `radian`。如果未来切到官方 `sensor_msgs/JointState`，
仍必须按已核实的 `name` 映射到 J1--J6，缺名或乱序无法确认时整帧 invalid。
当前源码中 TCP TF 查询和 Recorder stale fallback 的默认频率都是 5 Hz，不能据此
假设满足 10 Hz 采样；第一次真机采集必须让 preflight 实测并通过 joint rate 检查。

为避免手动移动或 IDLE 状态生成假 episode，Recorder 不会在 `active: false -> true`
后立即开档；它必须同时观察到 `OPEN_GRIPPER` 或 `MOVE_PRE_GRASP`，否则只等待真实
任务 phase。每条新 episode 开始时会快照当时新鲜的 `initial_joint_pos`、
`initial_tcp_pose` 和锁定苹果信息（camera/base xyz、id、confidence、bbox）。任务
`state_mapping` 是全局固定表并写入每个文件，不会按 episode 重新编号。默认
`joint_state_topic` 为 `/rm_json_bridge/arm_state`。

默认 `min_episode_samples=5`、`min_episode_duration_sec=0.5`；样本过少、持续时间过短，
或最终状态为空/`IDLE`/`UNKNOWN` 时，writer 会强制 `task_success/success=false`，并
在 `failure_reason` 与 QC 中记录原因，文件仍保留。

每个 dataset 第一维 T 一致、chunked/resizeable，图像使用 HDF5 LZF。attributes
包括 episode/start/end/duration/sample count/rate、相机型号/序列号/depth scale、
分辨率、单位/frame、state mapping、`task_success`、terminal/failure reason、
`physical_grasp_success=-1`、软件与 dataset version。

## 8. 当前仍采不到的数据

- EG2 当前启用 `realman_builtin` 路径，没有经过真机验证的实时位置、力、电流、
  moving 查询；所以 measured state 为 unknown/NaN，command 单独保存，绝不冒充
  实测状态。
- 没有物理“苹果确实被夹住”的传感器或人工标签；`task_success` 仅表示现有状态机
  正常完成。
- teacher bbox/confidence 只保存本次锁定目标，不是所有候选苹果的完整检测表。
- ROS 图审计时业务节点未在线；首次真机验证必须复核实际 QoS/type/数据 age。

## 9. ACT / Behavior Cloning 训练前

先采集多目标位置、光照、遮挡和失败 episode；为每条 episode 补充人工
`physical_grasp_success`；用 inspector/replay 剔除低 valid ratio 或错位样本；按
episode 而非 frame 划分 train/validation/test；统计 joint/TCP/action 归一化量；
确认 TCP 使用 controller `robot_end` 还是 `gripper_tip_center` 作为 observation、
reward 和 expert-trajectory frame；第一版 SAC 的 policy action frame 是六轴 joint
space。为 BC/ACT/Diffusion Policy 决定 expert action chunk/horizon，并明确夹爪离散
动作与 no-op mask。完成离线评估前，不把任何学习策略接入真机控制 service。

## 10. 半自动真机批量采集

本阶段在 Recorder 之外增加了独立 `rl_collection_orchestrator`。Recorder 仍然
只订阅、缓存和写盘；调度器才负责在双重环境变量门控后调用现有、人工验证过的
service。它不实现轨迹、逆解、工作空间或安全限位。

```text
人工放置苹果
  -> least-sampled init pose 规划
  -> Recorder 暂停（只暂停记录）
  -> 已验证初始化 service
  -> RGB/depth/RobotState/target 新鲜度检查
  -> 发布一次性 initialization metadata
  -> Recorder 恢复
  -> /pick_task_manager/start
  -> 等待 active true -> false 与 HDF5 finalized
  -> QC + manifest
  -> 人工物理标签或 label later
```

初始化动作发生在 Recorder 恢复之前，因此不会混入专家轨迹。episode 第一段来自
初始化后的真实状态；初始化关节、TCP、pose id、seed、实际 offset 与时间写在
episode metadata 中。

### 10.1 四个终端

Terminal A（同一 D435i pipeline，不能再由 GUI 启动第二个相机进程）：

```bash
CAMERA_SERIAL=908212070350 \
  /home/nvidia/RL-Orchard/scripts/run_rl_vision.sh
```

Terminal B（只启动现有 instrumented arm stack，脚本自身不调用动作）：

```bash
ORCHARD_ARM_ENABLE=YES \
  /home/nvidia/RL-Orchard/scripts/run_instrumented_arm_stack.sh
```

Terminal C：

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_recorder.sh
```

Terminal D 先做纯规划 dry-run：

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_collection.sh --dry-run --plan-only
```

再做 ROS 图只读 dry-run（创建订阅者，但不调用 Home/start/motion/gripper）：

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_collection.sh --dry-run
```

人工确认相机、急停、工作空间、Home、夹爪和配置后，才可显式打开两个门控：

```bash
ORCHARD_ARM_ENABLE=YES \
ORCHARD_COLLECTION_ENABLE=YES \
  /home/nvidia/RL-Orchard/scripts/run_rl_collection.sh --episodes 1
```

任一变量缺失或不是精确的 `YES` 都强制进入 dry-run。`--dry-run` 的优先级更高，
即使两个变量存在也不会调用动作 service。

### 10.2 初始化 pose library 与安全默认值

配置位于 `src/orchard_picker/config/rl_collection.yaml`。出厂内容只有：

```yaml
initialization:
  mode: pose_library
  initial_pose_library:
    - id: home
      enabled: true
      verified: true
      motion: home_service
      service: /pick_task_manager/home
  allow_verified_movej_service: false
  bounded_random_jitter:
    enabled: false
    joint_offset_min_deg: [0, 0, 0, 0, 0, 0]
    joint_offset_max_deg: [0, 0, 0, 0, 0, 0]
```

默认不会产生随机目标。多个白名单 pose 使用 manifest 中计数最少者；并列时用
`selection_seed + episode index` 做可复现选择。

额外 pose 推荐由人工验证后的 `Trigger` service 提供。MoveJ entry 只有在填写六个
关节、正速度、`verified: true` 和一个会执行项目安全检查的 service 后才可配置；
调度器明确拒绝 `/rm_json_bridge/movej`，避免绕过 `pick_task_manager` 的轴端高度等
检查。当前 overlay 没有擅自新增任意位姿运动接口，因此第一次真机采集只支持
现有安全检查过的 Home。以后存在经验证的安全 MoveJ service 后，才能同时显式设置
`allow_verified_movej_service: true`、非零 bounds 和 `enabled: true` 开启 bounded
jitter。

### 10.3 苹果位置 bin 与覆盖平衡

`apple_position_bins.enabled` 默认 `false`，没有硬编码真实工作空间边界。人工填入
x/y/z boundaries 与每轴 `N+1` 个 label 后，episode 和 manifest 会保存组合 bin。
可在 `suggested_regions` 填入希望覆盖的组合名称；调度器只提示当前计数最少区域，
不会控制任何外部设备移动苹果。

### 10.4 人工物理结果与 label later

正常交互会询问：

```text
Physical grasp success? [1] success [0] failure [u] unknown [l] later
```

`task_success` 始终表示既有状态机结果；`physical_grasp_success` 只来自人工标签，
编码为 `1/0/-1`。失败原因是自由字符串，工具不限制枚举；`operator_note` 也保存。
连续采集可使用：

```bash
/home/nvidia/RL-Orchard/scripts/run_rl_collection.sh --label-later
```

之后逐条标注：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/label_episode.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5
```

也可非交互执行：

```bash
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/label_episode.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5 \
  --label 0 --failure-reason apple_slip --operator-note "slipped during retreat"
```

失败、中止和 safety stop 文件都保留，不会因标签为 0 而删除。

## 11. 现有 demonstration phase mask 与固定末端接近

本节记录的是当前 `apple_rl_v1` 中基于 TCP transition 的历史 demonstration mask，
不是未来第一版 SAC 的 policy action 定义。字段名 `valid/policy_action` 为兼容旧
HDF5 保留；真正 SAC action 已在第 14 节统一为 delta-joint。当前 Recorder 已增量
记录 Expert 的 `executed_delta_joint`，但不会写入在线 SAC rollout 专用 action。

原 `apple_rl_v1` dataset 全部保留。新增的 `task/control_mode` 固定 enum 为：

| phase | control mode | `valid/policy_action` |
|---|---|---|
| `MOVE_PRE_GRASP` | `LEARNABLE_POLICY` | 同时要求 `valid/action=true` |
| `MOVE_GRASP` | `SCRIPTED_FINAL_APPROACH` | false |
| `OPEN_GRIPPER`、`CLOSE_GRIPPER`、`RELEASE_GRIPPER` | `SCRIPTED_GRIPPER` | false |
| `GRASPED_HOLDING`、`TWIST_J6`、`RETREAT`、`HOME` | `POST_GRASP_SCRIPTED` | false |
| 其他/未知/终止 phase | `NON_POLICY` | false |

所以对当前基于 TCP expert trajectory 的离线筛选可使用：

```python
policy_rows = episode["valid/policy_action"][:] == True
```

这里特意不把 `OPEN_GRIPPER` 的离散夹爪 command 当成现有 expert-motion 监督。
所有 phase 的
RGB-D、状态、真实 `executed_delta_tcp` 和专家 command 仍完整保存。

`scripted_final_approach_distance_m` 不是从配置中的“约 8 cm”复制，而是从结构化
`MOVE_PRE_GRASP` 与 `MOVE_GRASP` 两个 expert `target_tcp` 的平移目标计算。缺少任一
目标时保存 NaN，绝不伪造 0.08。

## 12. 扩展 HDF5、QC 与 manifest

新增内容保持向后兼容：

```text
episode_meta/{initial_joint_pos,initial_tcp_pose,initialization_offset}
events/expert_*                      # 独立 E 维，完整结构化 command event
task/control_mode
valid/policy_action
time/{rgb_age_ms,depth_age_ms,robot_state_age_ms,rgb_depth_delta_ms}
labels/                              # attrs，人工标注后存在
```

root attrs 增加 init pose/seed/mode/time/offset、初始苹果 snapshot、position bin、
control enum mapping、`scripted_final_approach_distance_m`、QC 和人工标签。旧路径没有
rename/delete，旧 episode 缺少新增字段时 inspector、replay 和 coverage 会按 legacy
处理。

`teacher/*` 仍是与 observation 对齐的 T 维最近 command；若 10 Hz 两帧之间发生
多个 command，全部原始结构化事件还会按 E 维写入 `events/expert_*`，因此不会因
采样降频丢失 pre-grasp/grasp target，末端固定接近距离也优先由完整 event 流计算。

每次正常 finalize 后自动检查并保留文件：

- RGB/depth/robot 是否存在及是否全黑/全零；
- 所有时序 dataset 的 T 是否一致、timestamp 是否严格递增；
- joint/TCP/action NaN/Inf、TCP quaternion norm；
- target/expert action 是否至少有效一次；
- `dones` 和末行 `valid/action=false`；
- phase 是否倒退、首帧是否已经晚于 approach；
- RGB-depth 时间差与记录的 robot state age。

问题写为 `quality_pass=false` 与 `quality_reason`，不删除 episode。异常退出仍保留
`.tmp.hdf5`。

`rl_dataset/collection_manifest.csv` 使用文件锁和临时文件原子替换，字段包括：

```text
episode_id,file,start_time,duration,sample_count,
initial_pose_id,initial_joint_summary,initial_tcp_summary,
initialization_mode,initialization_seed,apple_position_bin,
apple_base_x,apple_base_y,apple_base_z,target_id,target_confidence,
task_success,physical_grasp_success,terminal_state,failure_reason,operator_note,
rgb_valid_ratio,depth_valid_ratio,robot_state_valid_ratio,target_valid_ratio,
action_valid_ratio,policy_action_ratio,scripted_final_approach_distance,
policy_segment_rgb_valid_ratio,policy_segment_depth_valid_ratio,
policy_segment_all_valid_ratio,
quality_pass,quality_reason
```

`policy_segment_*` 只统计 `valid/policy_action=true` 的 approach 行：RGB/depth
分别统计对应有效率，`all` 同时要求 RGB、depth、robot state 和 apple target 有效；
没有 policy 行时三者为 `0.0`。覆盖率工具也会汇总这三个指标。

## 13. 检查、回放与覆盖率

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh

python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/inspect_rl_dataset.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5

python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/replay_episode.py \
  /home/nvidia/RL-Orchard/rl_dataset/episode_000001.hdf5

python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/report_collection_coverage.py \
  /home/nvidia/RL-Orchard/rl_dataset
```

coverage 只报告 episode/标签/QC 数量、init pose 与 apple bin 分布、valid ratio、
平均时长、phase/control-mode 样本数和 failure reason；它不会声称某个 episode 数量
足以训练 SAC/BC/ACT/Diffusion Policy。

纯离线三类 mock 与完整测试：

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python -m unittest discover -s /home/nvidia/RL-Orchard/tests -p 'test_*.py' -v
python /home/nvidia/RL-Orchard/tests/create_mock_episode.py \
  --output-root /tmp/apple_rl_mock
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/report_collection_coverage.py \
  /tmp/apple_rl_mock
```

## 14. 强化学习控制逻辑（SAC / delta-joint）

本项目未来第一版 SAC 的动作定义已经正式固定为六轴关节增量，而不是苹果坐标、
TCP 目标或 TCP delta：

```text
RGB-D / visual state
        │
joint position ── TCP pose ── target state ── robot state
        ▼
    Observation s_t
        ▼
     SAC Actor
        ▼
 normalized action [-1,1]^6
        ▼
 Joint Action Scaler
        ▼
 Δq1 ... Δq6
        ▼
 Safety Filter
        ▼
 q_target = q_current + Δq_safe
        ▼
 RealMan Joint Executor
        ▼
 Robot / next observation
        ▼
 reward / transition
```

正式 action space：

```python
Box(low=-1.0, high=1.0, shape=(6,))
```

并按每个关节独立缩放：

```text
delta_joint[i] = action[i] * max_joint_step[i]
q_target = q_current + delta_joint
```

本仓库不会猜测 `max_joint_step`、joint limit、速度或加速度；它们必须由人工审核
后传入未来环境配置。`q_target` 也不是未经 Safety Layer 检查就直接发送的命令。

选择 delta-joint 的工程原因是：动作局部、每步容易限幅、连续控制更平滑、适合 SAC、
容易从 joint demonstration 得到标签，并且不需要额外把 TCP action 隐式转换成多解
逆运动学。absolute joint target 不是永远禁止，只是第一版不采用。第一版也不采用
torque/current/PWM，因为当前系统没有经过验证的低层实时接口和安全验证链路。

### 现有 RealMan 接口审计

当前 overlay 中：

- `/rm_json_bridge/movej` (`MoveJ`) 接收 6 个 degree joint target、速度、blend 和
  `wait`，执行 blocking 的点到点 joint trajectory；
- `/rm_json_bridge/movej_p` 和 `/rm_json_bridge/movel` 是 Cartesian pose 接口，
  由当前传统抓取 expert 使用；
- `/rm_json_bridge/get_arm_state` 和 `/rm_json_bridge/arm_state` 提供当前关节、TCP
  pose 和 error telemetry；
- 当前自定义 bridge 没有暴露已经验证的 servo-joint、streaming joint target 或
  non-blocking cyclic joint API。RealMan 官方资料另有 `movej_follow` 与
  `movej_canfd`；后者是 joint-angle CANFD pass-through，高跟随模式需要不超过
  10 ms 连续周期，但本机没有发现/验证对应 driver 或 topic，不能假装它已可用。

因此现有 MoveJ 是未来 q_target 的“最接近接口”，但不能直接假设它适合作为 10 Hz
SAC action executor。未来需要增加或接入经过安全验证的连续小步 joint executor，
明确周期、延迟、队列、状态反馈、停止和 error semantics。当前 MoveJ/MoveL 流程
继续是 Expert / Teacher、demonstrator、baseline 或 fallback。

纯软件接口位于：

```text
src/orchard_picker/src/orchard_picker/rl_control/
  action_space.py       # normalized action -> delta_joint -> q_target
  safety_filter.py      # 独立 safety boundary，无机器人 I/O
  interfaces.py         # FutureJointSACEnv / JointControlExecutor contract
```

这些模块不启动 ROS、不创建 socket、不调用 MoveJ/MoveL/夹爪/Home。详细定义见
[docs/rl_control_architecture.md](/home/nvidia/RL-Orchard/docs/rl_control_architecture.md)。

### 与现有采集数据的关系

现有 `actions/executed_delta_tcp` 不删除、不重命名、不改含义；它是当前真实 TCP
状态变化和 expert trajectory 辅助信息，不再被描述为第一版 SAC policy action。
当前 Expert demonstration 已增量加入 `executed_delta_joint`、joint source/age、
command source 和 commanded joint target validity；旧 TCP transition、legacy
`valid/policy_action` 与旧文件含义保持不变。未来 SAC rollout 的
`normalized_action/requested_delta_joint/safe_delta_joint` 仍不写入 Expert episode。

RealMan 运行时只读审计见
[docs/realman_runtime_audit.md](/home/nvidia/RL-Orchard/docs/realman_runtime_audit.md)。
## 14. RL-Orchard 数据采集 GUI

真机批量采集优先使用独立副本 `/home/nvidia/orchard_experiment_rl_collect`，
而不是修改原 `/home/nvidia/orchard_experiment`。该 GUI 保持原有相机、机械臂和抓取操作，
只增加 Recorder 记录开关；Recorder 门控、急停和 invalid episode 规则见
[`docs/data_collection_usage.md`](docs/data_collection_usage.md) 的“RL 数据采集 GUI”章节。

GUI 的完整启动和采集步骤见
[`docs/rl_gui_usage.md`](docs/rl_gui_usage.md)。
