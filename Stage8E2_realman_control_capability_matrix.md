# Stage8E-2 RealMan 控制能力矩阵

审计范围：Origin `/home/nvidia/RL-Orchard`、真实控制器只读配置和 RealMan 官方接口资料。没有调用任何运动、servo、夹爪或配置接口。

| 功能 | 当前状态 | 是否满足 RL |
|---|---|---|
| TCP state query | 已实现 `/rm_json_bridge/get_arm_state`；request-response，Stage8E-1 实测约 `1.98 Hz`，仅查询成功后发布 `RobotState` | **否**：不能作为 20 ms feedback stream |
| UDP feedback | 控制器配置 `enable=true, cycle=1`，理论配置周期 `5 ms`；目标 `192.168.1.10:8090` 不属于 Origin；本地无 UDP receiver | **当前否**；协议能力存在，Origin 链路未建立 |
| UDP joint/pose/error | 官方 UDP state 结构包含 joint、waypoint、err、joint_err_code；当前 `custom.arm_current_status=false`，且未收到数据 | **UNKNOWN**，不能把配置推断为实际可用字段 |
| UDP controller timestamp | 官方状态结构审计未见控制器 timestamp 字段；只能可靠记录 host receive time | **否**，需定义时间对齐语义 |
| Movep_CANFD | 官方 JSON/Python/C/C++ API 存在；当前代码没有固定 schema、SDK 或 writer | **理论可行，当前否** |
| Movep_CANFD pose | 当前 work/tool 下的目标位姿；Euler 6 项或 quaternion 7 项；位置精度 `0.001 mm`，姿态 `0.001 rad` | **待确认**：Origin reported pose 原点仍 unresolved |
| Movep_CANFD high-follow | 官方要求周期不超过 `10 ms`；Gen3 有线网口资料给出理论最快 `2 ms` | **当前否**：没有稳定 100 Hz writer |
| Movep_CANFD 20 Hz 直发 | 20 Hz 为 50 ms，不满足 high-follow 的 ≤10 ms 要求；低跟随的实际版本/网络行为未在本机验证 | **否**，不能直接作为 high-follow RL loop |
| MoveJ_CANFD | 官方为关节角透传，不是 Cartesian action；当前无 writer | **不适用于当前 Cartesian action；当前未实现** |
| servo interface | 当前 `src/orchard_picker` 无 servo writer；没有官方 SDK/ROS driver overlay | **否** |
| Cartesian velocity pass-through | 官方 `rm_set_movev_canfd_init`/`rm_movev_canfd` 存在，单位 m/s、rad/s，示例周期 10 ms；当前无实现；Actor action 语义尚未确认 | **理论可行，当前否** |
| command frequency | 当前仅传统 service 调用；无非阻塞周期 writer、时间预算或 watchdog | **否** |
| feedback frequency | TCP query 实测约 2 Hz；UDP 目标错位，未测到 native push | **否** |
| pose semantics | active tool/work 名称已知，但 `get_current_arm_state.pose` 的物理原点和 Euler convention 未确认 | **否** |
| safety/watchdog | 现有 bridge 有 motion lock 和阻塞等待；没有 RL command watchdog 或 stale-feedback gate | **否** |

## 结论

RealMan 官方协议层面存在形成高速 Cartesian 闭环的接口，但 Origin 当前没有满足 RL 部署所需的 **native feedback receiver + 固定周期 Cartesian writer + watchdog + 已确认 pose/frame 语义**。因此本阶段结论是：

> **硬件/协议具有理论能力；当前 Origin 运行时不具备可验证的 RL 高速闭环条件。**

官方依据：

- [RealMan pose CANFD documentation](https://develop.realman-robotics.com/en/robot4th/json/motionConfig/)
- [RealMan Python pose CANFD API](https://develop.realman-robotics.com/en/robot4th/apipython/classes/movePlan/)
- [RealMan UDP active reporting](https://develop.realman-robotics.com/en/robot4th/json/udpConfig/)
- [RealMan ROS2 driver UDP parameters](https://develop.realman-robotics.com/en/robot4th/ros2/driver/)
