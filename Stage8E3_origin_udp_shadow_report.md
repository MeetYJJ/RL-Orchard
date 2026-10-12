# Stage8E-3 Origin UDP Feedback Shadow Runtime Validation

## 1. 执行范围

本轮在 Origin NVIDIA 工控机执行：

- Origin：`192.168.1.100`，接口 `eth1`；
- RealMan controller：`192.168.1.18`；
- 接收地址：`192.168.1.100:8090`；
- 只读绑定 UDP socket，未发送任何 UDP 数据；
- 没有启动 `rm_json_bridge`、`pick_task_manager`、`rl_collection_orchestrator`、command writer 或完整抓取流程。

原始包和统计保存于 [`/tmp/Stage8E3_udp_feedback_raw.json`](/tmp/Stage8E3_udp_feedback_raw.json)。

## 2. TCP 与 UDP 的区别

TCP `get_current_arm_state` 是请求-响应诊断接口。Stage8E-1 的低频采样约 2 Hz，只有客户端发送查询后才得到一条状态。

本轮 UDP 是控制器主动推送，Origin 只接收 JSON datagram。它不需要 Origin 发起状态查询，也没有使用 TCP bridge 的 blocking lock。UDP 反馈因此可以作为高速 telemetry 候选，但它仍然需要独立解析、host monotonic 时间、丢包和 stale 检查。

## 3. UDP 接收结果

正式采样 5 秒：

| 项目 | 结果 |
|---|---|
| packet count | `1001` |
| source | `192.168.1.18:42620` |
| destination | `192.168.1.100:8090` |
| packet size | `469–471 bytes`，中位数 `470 bytes` |
| JSON malformed | `0` |
| socket timeout | `0` |
| unique source | 1 |
| controller timestamp | UNAVAILABLE |

首次 10 秒裸 socket 监听也收到 `2001` 包，确认这不是单个偶发 datagram。当前 controller 的目标配置仍显示为 `192.168.1.10:8090`，但实际网络上已有包到达 Origin 的 `192.168.1.100:8090`；本轮没有修改该配置，也没有调用 setter。

## 4. 反馈频率和 latency 证据

- 实际频率：`200.200 Hz`；
- 接收间隔均值：`4.999996 ms`；
- 中位数：`4.996746 ms`；
- P95：`5.077873 ms`；
- 最大间隔：`5.255004 ms`。

UDP 状态包没有 controller send timestamp，不能从本轮数据计算真正的 controller-to-host absolute latency 或 RTT。上面的时间是 Origin `time.monotonic()` 接收时间间隔，反映到达 jitter。由于协议包没有 sequence number，精确丢包数不可证明；按 5 ms 期望周期估计丢包为 0，记录为 `ESTIMATE_ONLY_NO_SEQUENCE_NUMBER`。

## 5. 原始字段与 RobotState 映射

UDP JSON 中已观察到：

- `joint_status.joint_position`：6 个原始关节值，按 `0.001°` 转换为 rad；
- `joint_status.joint_err_code`：6 个关节错误码；
- `joint_status.joint_en_flag`：6 个使能标志，本轮全为 1；
- `err:[0]`：映射为 `sys_err=0` 和 `RobotState.error_code=0`；
- `waypoint.position/euler/quat`：原始 controller reported pose；
- `arm_err`：UDP payload 没有独立字段，保持 `null`；
- `controller_timestamp_s`：保持 `null`；
- `frame`：保持 `UNKNOWN`；
- `tcp_pose`：保持 `null`，因为 pose 原点仍为 `POSE_ORIGIN_UNRESOLVED`。

新增只读解析模块：[`udp_feedback_reader.py`](/home/nvidia/RL-Orchard/src/orchard_picker/src/orchard_picker/realman_adapter/udp_feedback_reader.py)。

该模块只有 UDP receive/decode/statistics 路径，没有 `send`/`sendto`、TCP client、运动 service 或 writer。精确 packet loss 需要控制器提供 sequence number；当前模块只提供基于期望周期的估计，并明确标注其限制。

## 6. Observation 完整性

UDP feedback 已经可以稳定提供关节、使能和原始 waypoint，因此解决了 Stage8E-2 中“Origin 完全收不到 native feedback”的问题。但本轮仍不能构造训练策略所需的完整 observation：

- policy TCP 的 pose origin 未确认；
- waypoint frame 未由 UDP 包声明；
- controller timestamp 不存在；
- apple/goal 字段没有进入本轮 observation；
- 视觉到 policy base frame 的变换仍未作为 shadow 输入证明；
- 任务 phase/hold 等字段不存在。

因此不能用零值补齐目标或 pose，也不能把 waypoint 未确认地命名为 `tcp_pose`。

## 7. Actor checkpoint 审计与 shadow 状态

本机现有冻结权重均为：

```text
actor_class: SACActor
architecture: sac_actor_mlp_v1
state_dim: 12
action_dim: 6
action: normalized joint-delta
```

例如：

- `training/runs/bc_pretrain_full_20260922_132554/actor_best.pt`
- `training/runs/bc_pretrain_full_20260922_163124/actor_best.pt`
- `training/runs/bc_pretrain_full_20260922_164435/actor_best.pt`

没有找到与当前任务声明一致的 **ReachActor 15D observation / 6D Cartesian action** checkpoint。现有 checkpoint 的 normalization 也是 12 维，且仓库文档定义的是 joint-delta action。使用它们会改变策略输入和动作语义，因此本轮没有强行加载到 15D observation，也没有伪造目标字段。

结果：

- UDP receiver：**PASS**；
- UDP → RobotState mapping：**PASS（pose/frame 字段保守标记）**；
- 15D ObservationBuilder：**BLOCKED**；
- ReachActor shadow inference：**NOT RUN**，因为没有匹配 checkpoint 和完整 observation schema；
- MockCommandWriter：**未启动**，没有生成或发送任何真实 command。

## 8. 当前阻塞项

1. 需要提供真正的 ReachActor 15D checkpoint、模型定义和 normalization metadata。
2. 需要冻结 15D observation 的字段顺序、单位、frame 和时间对齐规则。
3. 需要确认 UDP waypoint 的物理原点是 flange、robot_end 还是 active tool。
4. 需要确认 Actor 的 6D action 是 absolute pose、delta pose 还是 Cartesian velocity；现有仓库 checkpoint 是 joint delta。
5. 需要为视觉目标提供 policy base frame 下的位置和时间戳；本轮没有把视觉节点接入 shadow。
6. 需要 sequence number 或官方丢包语义，才能从估计丢包升级为精确丢包统计。

## 9. 安全确认

全程没有发送：

- `Movep_CANFD`；
- `MoveJ`、`MoveL`、`MoveJ_P`；
- servo；
- gripper；
- Home；
- stop；
- UDP 配置或 controller 参数修改。

本阶段证明了 **Origin 可以接收约 200 Hz 的真实 RealMan UDP feedback**。它还没有证明 15D ReachActor 可以运行，因为本机缺少匹配权重和完整 observation contract；在这些缺口补齐前，shadow inference 和任何 command writer 都保持关闭。
