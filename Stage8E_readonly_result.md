# Stage8E Origin RealMan 只读反馈结果

采集日期：2026-09-30  
采集位置：Origin  
采集模式：直接 TCP 固定 allowlist 只读查询  
样本数：8

## 1. 安全与连接范围

本轮只执行了以下查询：

- `get_arm_software_info`
- `get_joint_software_version`
- `get_current_tool_frame`
- `get_current_work_frame`
- `get_current_arm_state`

没有启动完整 ROS stack，没有使用 ROS `raw_json` service，没有运行 Actor 或 command writer。
没有发送运动、夹爪、stop 或配置命令。

原始查询响应见 [Stage8E_readonly_raw.json](/home/nvidia/RL-Orchard/Stage8E_readonly_raw.json)，反馈逐行证据见 [Stage8E_readonly_feedback.jsonl](/home/nvidia/RL-Orchard/Stage8E_readonly_feedback.jsonl)。

## 2. 控制器、版本和当前 frame

已确认：

| 字段 | 结果 |
|---|---|
| 控制器型号 | `RML63III-BI` |
| controller software | `V1.7.4`，build `2025/12/16 20:12:03`，commit `355fb1b` |
| plan software | `V1.7.4`，build `2025/12/16 20:12:21`，commit `c30d45c` |
| algorithm version | `1.5.9-0b8919e1` |
| dynamic model version | `2` |
| joint software raw version | `[54544,54544,54544,54544,54544,58640]` |
| protocol version | **UNAVAILABLE** |
| active tool | `Arm_Tip` |
| active tool transform | raw `pose=[0,0,0,0,0,0]`，`position=[0,0,0]`，`payload=0` |
| active work frame | `Base` |
| active work transform | raw `pose=[0,0,0,0,0,0]` |

按本地 `protocol.py` 的协议单位解释，tool/work 的平移和姿态均为零变换。该解释记录为查询结果，不修改任何 frame。

## 3. `get_current_arm_state` 原始字段

代表性原始响应：

```json
{"arm_state":{"err":[0],"joint":[2158,-84105,91237,88941,-88710,-51689],"pose":[-342453,-128042,597657,1542,-543,37]},"state":"current_arm_state"}
```

字段结论：

- `joint[6]`：已取得。原始值按本地协议转换为 JSONL 中的 radian 数值。
- `pose`：已取得并原样保存在 `controller_reported_pose.raw_pose`。
- `arm_err`：**UNAVAILABLE**。响应只有 `err`，没有名为 `arm_err` 的字段。
- `sys_err`：**UNAVAILABLE**。响应没有该字段。
- `controller_err_raw`：`[0]`。该值没有被擅自映射成 `arm_err` 或 `sys_err`。

## 4. 时间戳、频率和 RTT

控制器响应没有 controller timestamp。每行 JSONL 保存：

- `timestamp_s`：主机 `time.monotonic()`，发送查询前记录。
- `receive_timestamp_s`：主机收到完整 JSON 行的 `time.monotonic()`。
- `rtt_s`：发送到收到完整响应的主机单调时钟差。

8 次查询以约 0.5 s 间隔执行，没有持续加压控制器。结果：

| 指标 | 结果 |
|---|---:|
| 样本数 | 8 |
| 查询间隔 mean | 505.311 ms |
| 查询间隔 median | 505.064 ms |
| 查询间隔 P95 | 507.630 ms |
| 查询间隔 max | 507.905 ms |
| 低频查询频率 | 1.979 Hz |
| RTT mean | 5.186 ms |
| RTT median | 4.727 ms |
| RTT P95 | 7.321 ms |
| RTT max | 7.495 ms |
| timeout | 0 |
| malformed response | 0 |

`1.979 Hz` 是本轮低频查询的实际采样频率，不是控制器内部推送频率。当前代码没有发现经过验证的实时 UDP 或 ROS `/joint_states` feedback stream，因此控制器原生反馈频率为 **UNKNOWN**。

## 5. reported pose 语义

本地 bridge 和配置把该 pose 发布为 `base_link` 下的 `robot_end`，但本次 `get_current_arm_state` 原始响应没有携带物理原点或 frame 字段。当前 tool 为 `Arm_Tip`、work 为 `Base` 只能证明当前选择名称，不能证明 reported pose 一定是 flange、robot_end 或 active tool 原点。

因此本轮结论为：

```text
POSE_ORIGIN_UNRESOLVED
```

JSONL 不填 `tcp_pose`，只保留 `controller_reported_pose` 原始 pose。任何 Actor shadow observation 都不能直接把该字段当作 policy TCP。

## 6. 视觉反馈链

| 接口 | 可提供内容 | 单位/frame | 时间戳/频率 | 结论 |
|---|---|---|---|---|
| `/apple_positions` (`ApplePositionArray`) | 苹果三维位置列表 | m，`camera_link` | ROS header timestamp；相机配置30 Hz，实际未运行测量 | 位置可用性取决于视觉节点，当前未在线 |
| `/apple/target_point` (`PointStamped`) | 稳定苹果中心位置 | m，`camera_link` | ROS header timestamp；需连续5帧稳定，实际未测 | 只有位置，没有姿态 |
| `/apple/target_info` (`String`) | `camera_xyz_m`、id、confidence | m，`camera_link` | JSON内部无timestamp；实际未测 | 只有位置，没有姿态 |
| `/pick_task_manager/target_base` (`PointStamped`) | 任务管理器转换后的目标点 | m，`base_link` | 任务管理器生成 header；当前未运行 | 可作为软件转换后的 base 点，非当前实时证据 |
| `/apple/target_pose` | 目标 pose | 预期 base frame | 当前检测器源码没有发布该接口 | orientation **UNAVAILABLE** |

当前没有运行视觉节点或任务管理器，因此本轮没有真实 apple sample。视觉链无法在本轮证明能够提供实时 policy-base apple position；苹果姿态为 **UNAVAILABLE**。

## 7. 夹爪反馈

`/rm_json_bridge/arm_state` 的 `RobotState` 字段存在：

- `gripper_measured_state`
- `gripper_position`
- `gripper_force`
- `gripper_current`
- `gripper_moving`
- `gripper_measurement_valid`

但 bridge 在已读取 arm state 后明确填充：

```text
gripper_measured_state = -1
gripper_position = NaN
gripper_force = NaN
gripper_current = NaN
gripper_measurement_valid = false
```

当前没有经过验证的真实开度/位置反馈接口、单位、更新率或测量 timestamp。已有 gripper service 只表示发送目标命令，不能作为测量值。夹爪 aperture 为 **UNAVAILABLE**。

## 8. GraspLift 所需字段

| 字段 | 当前来源 | 语义 |
|---|---|---|
| phase | `/pick_task_manager/state` | 任务管理器维护的软件 phase，不是传感器测量；本轮未运行 |
| 初始苹果高度 | `/pick_task_manager/target_base.z` 可提供转换目标点 z | 软件变换后的目标点 proxy，不是独立真实高度传感器；本轮无值 |
| hold clock | 无独立接口 | 只有任务状态 header/软件 sleep 参数，没有真实 hold clock |

不能用 phase、目标点 z 或上次命令伪造 GraspLift 观测。

## 9. 15D Reach 与 31D GraspLift

### 15D Reach

**不能完整构造。** 当前仓库实际定义的第一版 RL state 是 12D：

```text
joint[6] + tcp_xyz[3] + goal_relative_xyz[3]
```

本轮还缺少已确认物理原点的 TCP pose，以及实时 policy-base 苹果位置。仓库也没有独立实现的 15D Reach schema，不能自行补齐三维字段。

### 31D GraspLift

**不能完整构造。** 缺少：

- 已确认 policy TCP pose；
- 实时 policy-base 苹果位置和姿态；
- 真实夹爪开度/位置反馈；
- 真实 phase/hold clock 语义；
- 仓库中的固定 31D GraspLift schema。

## 10. Movep_CANFD 静态审计

- 本地没有 `Movep_CANFD` 或 `movep_canfd` 实现、service、topic 或 SDK wrapper。
- 本地只在文档和数据枚举中出现 `movej_canfd` 名称，没有可调用入口。
- `rm_json_bridge` 当前只暴露传统 MoveJ、MoveJ_P、MoveL 和夹爪服务。
- 没有固定的非阻塞周期 command schema。
- 没有已实现的 watchdog、命令超时和反馈闭环。
- 官方资料描述的是 `movej_canfd`，不能把它等同于不存在的 `Movep_CANFD`。

本轮没有调用任何 CANFD 或 motion 接口。

## 11. 阻止 Actor shadow mode 的具体缺口

1. reported pose 的物理原点和 policy TCP 语义未确认。
2. controller timestamp、独立 arm_err、sys_err 未提供。
3. 当前只有低频请求采样，没有经过验证的实时反馈流。
4. 视觉节点本轮未在线，policy-base 苹果位置没有实时证据。
5. 苹果姿态接口不存在，记为 UNAVAILABLE。
6. 夹爪只有命令接口，没有真实 aperture measurement。
7. GraspLift phase、初始高度和 hold clock 没有真实测量语义。
8. 15D Reach/31D GraspLift observation schema 尚未固定。
9. 本地没有 RL 专用非阻塞关节 command writer 和 watchdog；本轮保持关闭。

## 12. 文件与最终安全确认

- JSONL： [Stage8E_readonly_feedback.jsonl](/home/nvidia/RL-Orchard/Stage8E_readonly_feedback.jsonl)，8 行。
- 原始查询： [Stage8E_readonly_raw.json](/home/nvidia/RL-Orchard/Stage8E_readonly_raw.json)。
- 报告： [Stage8E_readonly_result.md](/home/nvidia/RL-Orchard/Stage8E_readonly_result.md)。

全程没有发送任何 MoveJ、MoveJ_P、MoveL、Movep_CANFD、Movej_CANFD、servo、gripper、stop 或配置命令，也没有启动 Actor command writer。

官方 frame 查询协议参考：[RealMan JSON Frame Command Set](https://develop.realman-robotics.com/en/robot/json/coordinate/)。
