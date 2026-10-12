# Stage8E-6：ReachActor 真机部署路径重新评估

## 结论

本次修正是正确的。ReachActor 不需要苹果姿态，也不需要端到端视觉策略；它只需要一个与训练语义一致的目标 TCP pose。目标姿态可以由固定横向抓取姿态提供，视觉只需要提供目标位置（通常是苹果中心/抓取点的 XYZ）。

当前仍不能直接进入 ReachActor shadow inference。原因不是苹果姿态缺失，而是以下四项还必须对齐：

1. UDP waypoint 到 policy TCP 的精确变换；
2. 目标 XYZ 的 frame、单位和时间戳；
3. 固定抓取姿态与训练数据中的 goal orientation 是否完全一致；
4. RealMan 六轴顺序与训练模型 joint 顺序。

因此当前状态为：`REACH_OBSERVATION_BLOCKED`，但阻塞范围已经从“需要苹果姿态”收敛为“TCP/目标/frame 语义对齐”。

## 1. 修正后的 ReachActor observation

保持训练好的 15D contract：

```text
[joint_position_rad(6),
 tcp_xyz_m(3),
 target_xyz_m - tcp_xyz_m(3),
 rotation_error_rotvec_rad(3)]
```

其中：

- `joint_position_rad`：RealMan UDP joint position 转换后的六轴关节位置；
- `tcp_xyz_m`：当前 policy TCP 在 policy/base frame 下的位置；
- `target_xyz_m`：固定横向抓取姿态对应的目标 TCP 位置；
- `rotation_error_rotvec_rad`：固定目标姿态相对当前 TCP 姿态的旋转误差。

如果固定目标姿态为 `R_fixed`，当前 TCP 姿态为 `R_tcp`，则：

```text
R_error = R_fixed @ R_tcp.T
rotation_error = log_SO3(R_error)
```

因此只要 `R_fixed` 与 `R_tcp` 使用相同 frame 和旋转约定，就不需要苹果 orientation。不能把“横向”自行解释成新的姿态数值；`R_fixed` 必须与训练/teacher 数据的目标姿态一致。

## 2. 目标 TCP 的生成链

训练机已有的目标几何关系是：

```text
grasp_xyz = apple_base_xyz + grasp_offset_base
pregrasp_xyz = grasp_xyz - approach_vector_base * approach_distance
```

其中 `approach_distance=0.08 m`，held orientation 生成 approach vector；Grasp 与 PreGrasp 共享 orientation。Stage8D/Stage6D 的 Direct-Grasp 验证使用了这条几何关系。

真机部署应固定为：

```text
vision apple/target XYZ
→ target grasp XYZ
→ fixed horizontal grasp orientation
→ target TCP pose in policy/base frame
→ 15D observation
→ ReachActor Cartesian action
```

注意：如果训练 checkpoint 实际使用的是每个 episode 的 held orientation 分布，就必须证明“固定横向姿态”属于同一训练 contract；否则不能只修改真机配置而声称输入一致。必要时应先在训练机用固定姿态回放验证，而不是直接真机测试。

## 3. 当前状态重新分类

### 不再是阻塞项

- 苹果 orientation；
- 苹果姿态估计；
- GraspLift 31D observation；
- 端到端视觉 RL planner；
- ReachActor 重新训练。

### 仍是阻塞项

- UDP `waypoint` 是否为 policy TCP，或需要固定 tool offset；
- waypoint 与目标 XYZ 是否在同一 base/work frame；
- `waypoint.position/euler/quat` 的运行时姿态约定；
- RealMan raw joint 六轴顺序、符号、单位与训练 joint 顺序；
- 固定横向 `R_fixed` 的精确矩阵/四元数和来源；
- 视觉目标 XYZ 的 frame、单位、时间戳和 stale 规则。

## 4. 下一步最短路径

### 4.1 Origin1 只读资料补齐

请 Origin1 只读整理以下文件，禁止运动：

```text
/tmp/stage8e6_reach_observation_evidence/
```

必须输出：

1. `pose_semantics.json`
   - UDP waypoint 的原点：flange / robot_end / active tool / TCP / UNKNOWN；
   - frame：base / work / UNKNOWN；
   - position 单位；
   - Euler 单位和顺序；
   - quaternion 缩放、分量顺序和约定；
   - active tool transform；
   - active work transform；
   - 每项文件/函数证据。

2. `joint_mapping.json`
   - raw index 0…5 对应的 RealMan joint；
   - 单位、符号和零点；
   - 与训练 `joint1..joint6` 的对应关系。

3. `fixed_grasp_pose.json`
   - 传统控制实际使用的固定横向姿态；
   - rotation matrix、quaternion、Euler 原始值；
   - frame、单位和来源代码；
   - 是否与采集/训练 goal orientation 一致。

4. `target_generation.json`
   - 苹果/抓取点 XYZ 来源；
   - `grasp_offset_base`；
   - approach distance；
   - vision source frame；
   - 到 policy/base frame 的变换；
   - 目标时间戳和 stale 规则。

5. `udp_samples.jsonl`
   - 至少 3 秒只读 UDP 原始包；
   - host monotonic timestamp；
   - raw joint、converted rad、waypoint 原始字段；
   - `joint_err_code`、`joint_en_flag`、`err`；
   - 不得改 UDP 配置。

6. `observation_reconstruction_samples.jsonl`
   - 仅在以上语义已确认后生成；
   - current joint；
   - current policy TCP；
   - target TCP；
   - raw 15D observation；
   - normalized 15D observation；
   - finite/shape/frame checks；
   - 不运行 Actor，不发送 command。

7. `Stage8E6_origin_readonly_report.md`
   - 明确输出 `REACH_OBSERVATION_READY` 或 `REACH_OBSERVATION_BLOCKED`。

### 4.2 训练机离线回放

收到 Origin1 资料后，在训练机完成：

1. 用录制的 q、pose、target 重建 15D；
2. 对比 MuJoCo observation 的字段顺序、单位和归一化；
3. 使用冻结 ReachActor 做离线 shadow inference；
4. 检查 action bounds、NaN/Inf、姿态误差和时间对齐；
5. 不连接硬件，不发送命令。

### 4.3 Origin shadow

离线回放通过后，Origin 才运行：

```text
UDP feedback + target XYZ
→ ObservationBuilder
→ frozen ReachActor
→ SafetyGate
→ MockCommandWriter
```

仍不启用 Movep_CANFD writer。记录 observation、normalized observation、Actor action、延迟和 stale/error gate。

### 4.4 最小真机验证

只有以下条件全部满足后才另行授权：

- `REACH_OBSERVATION_READY`；
- 录制数据离线重放通过；
- shadow action 稳定且无越界；
- TCP/frame/tool 变换已确认；
- 固定姿态与训练 contract 一致；
- safety watchdog、workspace/joint margin、timeout 已审查。

第一动作仍应是单轴极小 Cartesian step，而不是 Initial→Grasp 全流程。该动作需要单独的 Stage6C/真机授权。

## 5. 对原思路的评价

“移除苹果 orientation、使用固定横向抓取姿态”是正确的范围收缩，能减少 observation 和部署依赖；但它不能自动解决 TCP 原点、目标 frame、关节顺序和训练姿态分布的一致性。最优策略是先完成只读 observation contract 对齐，再做 shadow，再做极小动作，最后才进入完整 ReachActor 控制。

本阶段不重新训练、不修改 ReachActor、不修改传统控制、不连接真实机械臂。

