# Stage8E-2 RealMan High Frequency Control Path Audit

## 1. 审计结论

控制器型号为 `RML63III-BI`，软件/计划版本为 `V1.7.4`。本轮只读审计没有发送任何运动、servo、夹爪、stop 或配置命令。

官方资料显示 RealMan 有位姿 CANFD 透传、关节 CANFD 透传和 UDP active reporting。当前 Origin 项目只有自定义 TCP JSON bridge 和传统 ROS service，未形成可运行的高速 RL 闭环。结论分为两层：

1. **协议理论能力：存在。** 官方 `movep_canfd` 能接收当前 work/tool 下的 pose，UDP 可按 5 ms 倍数上报状态。
2. **当前 Origin 能力：不满足。** 没有 native UDP 接收器、没有 Movep writer、没有时间戳对齐、没有 watchdog，reported pose 原点也未确认。

## 2. TCP 约 2 Hz 的原因

Stage8E-1 的约 `1.977 Hz` 来自低频 request-response 采样：每次发送 `get_current_arm_state`，等待一条 TCP JSON 响应，再等待约 `0.5 s`。它测量的是查询链路的样本间隔，不是控制器内部伺服频率。

本地实现的关键路径是：

- `src/orchard_picker/scripts/rm_json_bridge.py` 的 `handle_get_arm_state` 每次服务调用才发查询；
- `src/orchard_picker/src/orchard_picker/realman_json_client.py` 是带 socket/send/motion lock 的 TCP 行协议客户端；
- `RobotState` 在查询成功后旁路发布，不是独立硬件 feedback stream；
- `tcp_tf_publisher` 默认轮询 5 Hz，但它仍然依赖同一个服务查询，不代表控制器 push 频率。

因此普通查询不能直接用于 20 ms ObservationBuilder。提高查询压力也不能替代 native feedback，因为请求排队、TCP RTT、ROS service 调度和锁竞争都会进入控制周期。

## 3. Movep_CANFD 官方能力

官方协议定义的 `movep_canfd`：

- 目标是当前 **work frame 和 tool frame** 下的位姿；
- Euler 形式使用 `pose[6]`，四元数形式使用 `pose_quat[7]`；
- 位置精度 `0.001 mm`，姿态精度 `0.001 rad`；
- `follow=true` 为高跟随，透传周期要求不超过 `10 ms`；
- `trajectory_mode=0` 是完全透传，`trajectory_mode=1/2` 是曲线拟合/滤波模式；
- 控制器接收位姿后先做逆解；若解与当前关节接近，则可直接下发，跳过额外轨迹规划；
- 官方资料说明第二代普通 Wi-Fi/网口透传最快约 `20 ms`，高速网口可到 `10 ms`（需先启用配置），第三代有线网口理论可到 `2 ms`。

官方资料：[JSON pose CANFD](https://develop.realman-robotics.com/en/robot4th/json/motionConfig/)、[Python `rm_movep_canfd`](https://develop.realman-robotics.com/en/robot4th/apipython/classes/movePlan/)、[C/C++ pose CANFD](https://develop.realman-robotics.com/en/robot4th/apic/classes/movePlan/)。

### 对 RML63III-BI 的适用性

官方 release notes 明确列出 RML63III-B 的支持和 V1.7.3 的 Cartesian velocity pass-through 更新，但文档没有在本机审计中给出 `RML63III-BI + controller V1.7.4` 的逐项 API compatibility matrix。`RML63III` 是机械臂构型名称，不能单凭名称把控制器网络路径认定为 Gen3 2 ms。需要 RealMan 根据控制器序列号、API2 版本和固件确认 `rm_movep_canfd`/`movep_canfd` 的可用版本。

## 4. UDP Active Reporting

Stage8E-1 已读取控制器配置：

```text
enable = true
ip     = 192.168.1.10
port   = 8090
cycle  = 1
```

官方定义 `cycle` 的单位为 `5 ms`，所以 `cycle=1` 的理论配置周期是 `5 ms / 200 Hz`。官方 UDP state 结构可包含：

- `joint_status.joint_position`（0.001°）；
- `waypoint` pose；
- `err` system error；
- `joint_err_code`；
- `arm_current_status`。

当前配置的 `custom.arm_current_status=false`，所以不能把“接口支持这些字段”误写成“当前配置一定发送这些字段”。另外，官方公开的 UDP state 结构没有独立 controller timestamp 字段；部署端需要记录主机接收 monotonic time，并定义丢包、乱序和时间对齐策略。

Origin 本机地址为 `192.168.1.100`，控制器推送目标 `192.168.1.10:8090` 不是本机地址。当前项目也没有官方 API2 SDK、`rm_driver`、UDP socket receiver 或三线程 callback 接入。因此本机没有收到 native feedback。本轮没有调用 `set_realtime_push`，也没有监听并伪造 UDP 数据。

官方依据：[UDP JSON 配置](https://develop.realman-robotics.com/en/robot4th/json/udpConfig/)、[UDP state structure](https://develop.realman-robotics.com/en/robot/apic/struct/realtimeArmJointState/)、[UDP callback/API](https://develop.realman-robotics.com/en/robot/apic/classes/udpConfig/)。官方 Python 文档还要求三线程模式、正确的目标 IP/端口和未被防火墙阻断，才能收到 callback。

## 5. 当前代码接口审计

### 已有实现

- `rm_json_bridge.py`：TCP JSON 客户端、`get_current_arm_state`、传统 MoveJ/MoveJ_P/MoveL、夹爪和工具配置 service。
- `realman_json_client.py`：线程安全 TCP 行协议、响应等待、`_command_lock`/`_motion_lock`。
- `protocol.py`：joint/pose 的协议单位转换。
- `tcp_tf_publisher.py`：基于查询服务的低频 TF 发布。
- `RobotState.msg`：joint、pose、单一 `error_code` 和夹爪测量有效标志。

### 仅为设计/数据标签，不是执行接口

- `rl_data/schema.py` 中的 `MOVEJ_FOLLOW`、`MOVEJ_CANFD` 是数据事件枚举；没有对应 writer。
- 文档中的 `movej_canfd`、`movep_canfd`、官方 ROS topic 名称不是本工作区实现。
- `handle_raw_json`/`RawJson` 是通用任意 JSON service，不能当作固定 schema 的 RL command writer；本轮没有使用它。

### 明确不存在或未证明

- `Movep_CANFD` writer：不存在；
- `MoveJ_CANFD` writer：不存在；
- servo writer：不存在；
- UDP receiver：不存在；
- 官方 SDK/API2 接入：未找到；
- 非阻塞周期发送器：不存在；
- feedback watchdog/stale gate：不存在；
- controller timestamp：未提供。

## 6. RL 控制周期重新定义

20 Hz Actor action 是策略更新频率，不应直接等同于 command transport 或内部关节伺服频率。推荐部署目标如下，待官方确认和只读链路验证后实现：

```text
native UDP feedback       100~200 Hz，目标 5~10 ms 周期
ObservationBuilder        使用最新完整 frame，拒绝 stale/malformed
Actor inference            20 Hz，50 ms 策略更新
action hold/interpolation  100 Hz，10 ms command clock
Cartesian writer           稳定 100 Hz；高跟随时满足 ≤10 ms
robot internal servo       由 RealMan 控制器/驱动器负责，具体频率 UNKNOWN
watchdog                   丢 feedback、超时、错误或 writer deadline 即禁止新 command
```

如果 Actor 输出的是 delta pose，必须先定义它如何变成当前 work/tool frame 下的绝对 pose，再进入 `movep_canfd`。如果 Actor 输出的是 Cartesian velocity，官方另有 `rm_set_movev_canfd_init`/`rm_movev_canfd`：初始化参数含 frame type 和 `dt`，速度单位为 m/s、rad/s，文档示例以 10 ms 周期发送。当前仓库没有这条接口，且不能替 Actor 猜测 action 语义。见 [Cartesian velocity pass-through](https://develop.realman-robotics.com/en/robot/apipython/classes/movePlan/)。

因此：

- 20 Hz **策略**频率可以作为上层更新频率；
- 20 Hz **直接 high-follow Movep** 不满足官方 ≤10 ms 约束；
- command transport 至少应设计为稳定 100 Hz，再由策略输出 hold/interpolate；
- 在没有 native feedback 时，不允许用 2 Hz TCP query 假装完成闭环。

## 7. 当前阻塞项

1. UDP 推送目标不是 Origin，本机未收到 native feedback。
2. `custom.arm_current_status=false`，实际 UDP payload 字段尚未确认。
3. 没有官方 SDK/API2 或 `rm_driver` 接入。
4. 没有固定 schema 的 `movep_canfd` writer。
5. 没有非阻塞 100 Hz command clock 和 watchdog。
6. `get_current_arm_state.pose` 的物理原点、Euler convention 和 policy TCP 仍未确认。
7. 没有 controller timestamp，只能使用 host receive time。
8. Actor 的 6D Cartesian action 是 delta pose、absolute pose 还是 velocity，部署映射尚未冻结。
9. `RML63III-BI + V1.7.4` 的具体 API2/控制器兼容版本尚未由 RealMan 确认。

## 8. 下一步真实 shadow runtime 方案

仍保持不发送命令，按以下顺序推进：

### A. 只读 native feedback 接入

获取与 `V1.7.4` 匹配的官方 API2/ROS2 driver 版本说明，确认 RML63III-BI 支持矩阵。使用只读 `rm_get_realtime_push`/官方 callback 或等价只读 receiver，要求目标地址、端口和三线程模式明确；本阶段不调用 setter。

记录每包 host monotonic receive time、包解析错误、丢包/乱序、joint/pose/error/status 字段和实际 jitter。确认 `arm_current_status` 与 error 字段确实出现后，才进入 shadow observation。

### B. pose/frame 证据闭环

从官方协议和当前 tool/work 配置确认 reported waypoint 的原点；将 pose 语义与训练机 policy TCP 一一对齐。未确认前，ObservationBuilder 不得填 `tcp_pose`。

### C. dry-run command path

在不连接或不发送真实 writer 的条件下，固定 `TargetPose` schema、frame、单位、Euler/quaternion 选择、动作限幅、deadline 和 stale gate。用录制的 UDP feedback 做 shadow replay，统计 Actor 20 Hz 与 writer 100 Hz 的时间预算。

### D. 真机 shadow 后的动作前置条件

只有在 native feedback 连续、pose origin 已确认、动作语义已冻结、watchdog 已验证、writer 已通过静态/mock 审查后，才提出极小范围人工监督动作测试。该动作测试不属于本阶段，本轮没有进行。

## 最终判断

RealMan 协议具备支持 RL Cartesian 闭环的理论接口，但当前 Origin 只具备约 2 Hz 的查询式状态路径和传统阻塞式 bridge。当前不能声称已经具备高速 RL runtime，也不能用普通 TCP query 替代 native feedback。下一项最有价值的工作是获取匹配的官方 API2/driver 并在不改配置、不发运动命令的前提下完成 UDP native feedback shadow 接收验证。
