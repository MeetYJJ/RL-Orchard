# RL-Orchard 强化学习控制架构

本文定义未来第一版 SAC 控制的职责边界。它不改变当前已经验证的采摘状态机，
也不改变旧 HDF5 字段的含义；当前 Recorder 以向后兼容方式增加 Expert joint
transition 字段。当前 `orchard_picker` 的 MoveJ、
MoveJ_P、MoveL 和夹爪流程仍是 Teacher/Expert，不是 SAC 控制器。

## 1. 最终目标

未来策略学习的是高层、连续的六轴关节增量控制：在当前状态下，Actor 决定下一
个控制周期六个关节各自应该移动多少。RealMan 控制器负责可靠执行经过安全层
批准的 joint target；它仍然可以使用底层位置环、速度环和伺服器，但这些底层环
不应被描述为“RL 没有控制机械臂”。RL 控制的是 joint target 的决策层。

```text
RGB-D / visual state
        │
joint position ── TCP pose ── target state ── robot validity/phase
        ▼
     Observation s_t
        ▼
     SAC Actor
        ▼
 normalized action a_t ∈ [-1, 1]^6
        ▼
 JointActionScaler
        ▼
 Δq1 ... Δq6
        ▼
 JointActionSafetyFilter
        ▼
 q_target = q_current + Δq_safe
        ▼
 RealMan joint executor
        ▼
 robot / controller
        ▼
 q_(t+1), TCP_(t+1), next observation
        ▼
 reward, terminated, truncated, info
        ▼
 (s_t, a_t, r_t, s_(t+1), done)
```

本轮只实现 `JointActionConfig`、纯 Python scaler、纯 Python safety boundary 和
未来 environment/executor 的接口。它们不会创建 ROS node、socket、service client，
也不会发送任何机器人命令。

## 2. Observation：只确定概念，不冻结编码器

第一版暂不决定 CNN、RGB 分辨率、点云或最终扁平向量。未来 `s_t` 至少应能表达：

```text
visual_state       RGB-D 或未来视觉 encoder 的输出
joint_position     J1 ... J6，单位必须在环境配置中明确
tcp_pose           当前实际末端 pose，继续保留用于 observation/reward/debug
target_state       苹果目标的相关状态；具体是否使用 xyz 尚待实验决定
optional_robot_state error、phase、validity、连接/急停等必要状态
```

`target_state` 不等于 action。苹果坐标可以是观测、教师信息或 reward 辅助信息，
但第一版 SAC 的 action 不输出苹果 xyz 或 TCP target。

## 3. Action Space：六轴 delta joint

正式 action space 是：

```python
Box(low=-1.0, high=1.0, shape=(6,))
```

Actor 输出 `a_t` 后按每个关节独立缩放：

```text
delta_joint[i] = a_t[i] * max_joint_step[i]
q_target       = q_current + delta_joint
```

`max_joint_step` 必须由人工审核后通过配置提供。本 overlay 不填写实际角度、
速度、加速度或关节限制，也不把现有 Home pose 当成 RL action limit。当前纯软件
接口不猜测单位；未来环境应统一约定（推荐内部使用 radian）并在配置/metadata
中标明。

### 为什么第一版选择 delta joint

相对于 Actor 直接输出绝对 `q_target`，增量 action 更适合第一版工程实现：

1. 它天然表达连续控制中的局部状态变化；
2. 每一步可以直接受到 `max_joint_step` 限制；
3. 更容易做动作 clipping 和安全检查；
4. 小步更新更容易保持运动平滑；
5. 与 SAC 的连续 action space 直接匹配；
6. 可以从真实 joint trajectory 计算 demonstrator action；
7. 不需要 Actor 重新学习绝对姿态的全部坐标系偏置。

这不是说 absolute joint target 永远不能使用；只是第一版项目接口明确选用
`delta_joint`。

### 为什么不是 delta TCP

当前数据中 `actions/executed_delta_tcp` 仍然是完整真实轨迹的辅助状态变化和
expert trajectory 信息。它不再被定义为第一版 SAC 的最终 policy action。TCP delta
需要额外的逆解、奇异位姿处理、关节限位和多解选择；这会把 RL action 与另一个
隐含控制器耦合。第一版直接在 joint space 定义动作，避免把“TCP policy + IK”
误写成“RL 学习关节控制”。

当前 Recorder 还保存 `actions/executed_delta_joint[t] = q[t+1]-q[t]`，这是
Expert demonstration 的 joint-space transition label；它不是在线 SAC 的
requested/safe action。Cartesian expert command 没有真实 joint target 时，
`valid/commanded_joint_target` 必须保持 false。

### 为什么第一版不做 torque/current/PWM

当前设备、驱动和安全验证链路没有为本项目提供经过验证的 torque/current/PWM
控制接口。底层 RealMan 位置/伺服控制器已经负责低层闭环；第一版高层 SAC 使用
delta joint 即可。低层力矩控制需要新的实时接口、动力学、限幅、急停和安全验证，
不能从现有 MoveJ service 推断出来。

## 4. SAC Actor、Critic 与 Replay Buffer

Actor 学习：

```text
s_t -> a_t ∈ [-1, 1]^6 -> delta_joint
```

Critic 学习：

```text
Q(s_t, delta_joint_t)
```

也就是评价“在当前状态执行这一组六轴增量的长期回报”，Critic 不直接控制
机器人。

训练时概念数据流为：

```text
observation -> Actor -> delta_joint -> Safety Layer -> executor
           -> next observation + reward
           -> replay buffer
           -> Critic/Actor update
```

标准 transition 是：

```text
(s_t, a_t, r_t, s_(t+1), done)
```

其中 `a_t` 是真实 policy action（normalized action 或其明确的 delta-joint
表示，二者在训练接口中必须统一），不是底层 JSON 字符串，也不是 TCP delta。

部署时通常只需要：

```text
Observation -> Actor -> scaler -> Safety Layer -> q_target -> RealMan executor
```

Replay buffer 和 Critic 不需要进入实际机器人动作链路。

## 5. Safety Layer

RL Policy 不等于 Safety Controller。独立的 `JointActionSafetyFilter` 接收：

```text
current_joint
delta_joint_requested
```

并输出：

```text
delta_joint_safe
q_target_safe
valid
reason
```

未来安全层可以复用已验证的项目检查，并在接口中承载：

- normalized action / delta action clip；
- per-joint max step；
- joint limits；
- velocity/acceleration limits（只有真实限制已确认时）；
- robot state validity 和 error code；
- emergency stop；
- 已有 workspace constraint；
- 可选 collision constraint。

本轮纯模块只检查显式传入的限制，不创建新的机器人安全数值。未提供限制时不
伪造限制；未来整合必须把实际限制作为已审核配置或复用现有 safety implementation。
安全层拒绝动作时，环境应返回明确的 `info["safety_reason"]`，而不是静默发送。

## 6. RealMan low-level executor

当前源码审计结果：

| 接口 | 当前类型 | 语义 |
|---|---|---|
| `/rm_json_bridge/movej` | `orchard_picker/srv/MoveJ` | 6 个 degree joint target，速度/ blend，`wait`；blocking trajectory command |
| `/rm_json_bridge/movej_p` | `MovePose` | Cartesian pose point-to-point，当前专家 pre-grasp 使用 |
| `/rm_json_bridge/movel` | `MovePose` | Cartesian linear pose command，当前专家 grasp/retreat 使用 |
| `/rm_json_bridge/get_arm_state` | `GetArmState` | 读取 degree joints、pose、error |
| `/rm_json_bridge/arm_state` | `RobotState` | 旁路状态 telemetry，关节已转 rad |

`rm_json_bridge.py` 的 `MoveJ` handler 会把 degree 转成 RealMan 协议单位，clamp
速度到 `[1,100]`，clamp blend 到 `[0,100]`，然后等待 JSON ack/trajectory done（当
`wait=true`）。这说明它是可靠的 blocking point-to-point command boundary，
不是已经验证的 10 Hz streaming servo-joint interface。

当前没有发现：

- servo-joint ROS service；
- streaming joint target topic；
- non-blocking cyclic joint position API；
- 可确认控制周期、速度/加速度约束和 command latency 的高频接口。

RealMan 官方资料另有 `movej_follow` 与 `movej_canfd` 能力描述；其中
`movej_canfd` 是 joint-angle CANFD pass-through，高跟随模式需要不超过 10 ms
的连续目标周期。它们没有在本 overlay 中出现或验证，不能当作现成执行器。

另外，当前 `arm_state` 是在已有 `get_arm_state` 查询成功后旁路发布，并非驱动内部
独立的固定周期实时状态流；其实际频率取决于现有查询者。源码里存在 Home tolerance、
TCP/workspace/轴端高度检查和控制器 error 检查，但没有可直接复用为 SAC 配置的完整
六轴 joint-limit、per-step、velocity 或 acceleration 表，因此本文不抄写或猜测这些
数值。

因此未来 SAC 不能直接假设 `/rm_json_bridge/movej` 适合作为每个 10 Hz action 的
执行器。真正接入前需要增加或接入一个经过安全验证的连续小步 joint command
executor，至少明确 command period、non-blocking/queue semantics、状态反馈时序、
stop/timeout/error semantics 和 RealMan 控制器对 streaming joint target 的支持。
在此之前，现有 MoveJ 只能作为 expert、低频初始化或 fallback/reference，不能被
文档冒充为已经完成的 SAC action executor。

## 7. Environment 的未来 step contract

未来 Gym/Gymnasium 环境的 `step(action)` 概念顺序是：

1. 读取当前真实 `q_t`；
2. 校验 Actor action 为 `(6,)` 且在 `[-1,1]`；
3. scaler 计算 `delta_joint`；
4. safety filter 检查并计算 `q_target_safe`；
5. 通过独立 joint executor 发送 target；
6. 等待一个已定义的 control period / next-state event；
7. 读取 `q_(t+1)`、TCP 和 observation；
8. 计算 reward、terminated、truncated、info；
9. 返回 `obs_next, reward, terminated, truncated, info`。

`FutureJointSACEnv` 和 `JointControlExecutor` 只定义 Python contract，不实现
任何第 5 步 I/O；本轮 Recorder 的 joint transition 字段是向后兼容的增量扩展。

## 8. Reward：只定义候选项

Reward 尚未冻结。候选项包括：

- TCP 到 pre-grasp target 的距离改善；
- 成功/失败结果；
- collision 或 safety rejection；
- joint limit margin；
- action magnitude；
- smoothness；
- timeout。

例如距离进步可写成 `r_progress = d_t - d_(t+1)`，但当前不确定权重、终止条件或
真实物理抓取 reward，不在本轮写死数值。

## 9. Expert、数据采集和最后 8 cm

当前传统流程的新定位是：

```text
Traditional controller (MoveJ_P / MoveL / gripper)
        -> Expert trajectory
        -> demonstration dataset
        -> BC / ACT / initialization
        -> future SAC fine-tuning
```

现有视觉目标计算、TCP target、MoveJ_P、MoveL、夹爪和 Home 都保留为 expert、
baseline 或 fallback。它们不是第一版 SAC 的最终动作输出。

项目仍可以让 RL 负责到 pre-grasp，最后约 8 cm 由 scripted controller 负责；
`executed_delta_tcp` 保留为辅助状态变化和 expert trajectory 信息，不能冒充 SAC
action；`executed_delta_joint` 是当前 Expert 数据可用于 BC/offline RL 的候选标签。
