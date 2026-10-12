# Stage 4: RML63-B SAC+BC MuJoCo rollout

This is a MuJoCo-only initial-pose to pre-grasp runner. The MJCF uses the
RML63-B URDF and matching six link meshes from the installed
`rm_63_b_description` package; joint origins, axes, limits, inertial values,
and visual/collision geometry are carried into the model. `robot_end` is the
origin of the URDF's terminal `Link6` flange, matching the project's recorded
controller TCP frame. No gripper, fruit, grasp, retreat, ROS, or real robot is
loaded or controlled.

The original URDF used for the conversion is retained at
`assets/rml63b/rm_63_b_description.urdf`. `check_fk_consistency.py` computes
independent serial-chain FK from that URDF and compares Link1 through Link6
against the MuJoCo bodies before replay. It also reports the separately
configured project transform `robot_end -> gripper_tip_center = [0, 0,
0.168]` and a 0.158 m flange-local-Z candidate as diagnostics only; neither
is applied to `robot_end`.

The environment exposes `get_flange_pose()` separately from
`get_tool_pose(explicit_offset_xyz, explicit_offset_rotation)`, so a tool
transform cannot be silently introduced into the policy state.

The policy input remains
`[joint1..joint6, robot_end xyz, pregrasp xyz - robot_end xyz]`. The runner
uses the supplied BC `normalization.npz` for both state normalization and
action scaling. The action pipeline is `tanh(mean) -> action_scale -> delta_q
-> q + delta_q`, followed by an RML63-B joint-limit check and position servo.
The fixed MuJoCo physics timestep is 2 ms and the policy interval is 250 ms
(125 physics steps), matching the approximately 4 Hz collection interval. One
joint-delta action sets one target, which is held for all 125 physics steps.
The default 5 mm pre-grasp threshold matches
`EpisodeWriter.PREGRASP_REACHED_TOLERANCE_M` and can be overridden on the CLI.

## Run on the training workstation

Sync the Stage 4 source and model assets listed in the handoff, then run only
inside the training machine's `rl_orchard_mujoco` environment:

```bash
conda activate rl_orchard_mujoco
cd /home/ps/RL-Orchard
python training/mujoco/run_offline_sac_episode.py \
  --dataset-dir /home/ps/RL-Orchard/datasets/apple_rl_dataset_v1_20260922 \
  --split-manifest /home/ps/RL-Orchard/training/runs/bc_pretrain_full_20260922_211200/split_manifest.json \
  --normalization /home/ps/RL-Orchard/training/runs/bc_pretrain_full_20260922_211200/normalization.npz \
  --actor-checkpoint /home/ps/RL-Orchard/training/runs/offline_sac_bc_20260923_112947/actor_best.pt \
  --device cuda --max-episodes 3 \
  --pregrasp-distance-threshold-m 0.005
```

With no `--max-steps`, each SAC episode is capped at its own recorded count of
valid policy transitions. The Test split remains untouched for final
evaluation.

By default, up to the first three episode IDs from the manifest's held-out
Validation split are used; the Test split remains untouched for final
evaluation. Each HDF5 is opened read-only; only
`episode_meta/initial_joint_pos` and `episode_meta/pregrasp_pose_base` are
read. To choose specific episodes, pass `--episode-ids episode_... episode_...`.

The runner prints one compact JSON result per episode with the initial/final/
minimum goal distance, steps, success, joint-limit violation, and NaN/Inf flag.
MuJoCo runtime and rollout validation must be performed on the workstation;
this repository worktree's edge-computer environment is used only for static
checks.

## Expert replay consistency check

Run the FK diagnosis first. It does not load the Actor, modify HDF5, or run a
policy/expert action. A reset mismatch stops the command before replay:

```bash
conda activate rl_orchard_mujoco
cd /home/ps/RL-Orchard
python training/mujoco/check_fk_consistency.py \
  --dataset-dir /home/ps/RL-Orchard/datasets/apple_rl_dataset_v1_20260922 \
  --split-manifest /home/ps/RL-Orchard/training/runs/bc_pretrain_full_20260922_211200/split_manifest.json \
  --episode-id episode_000043
```

Before interpreting SAC rollout failures, run the raw recorded Expert replay
for one Validation episode. This does not load the Actor or normalization:

```bash
conda activate rl_orchard_mujoco
cd /home/ps/RL-Orchard
python training/mujoco/check_expert_replay_consistency.py \
  --dataset-dir /home/ps/RL-Orchard/datasets/apple_rl_dataset_v1_20260922 \
  --split-manifest /home/ps/RL-Orchard/training/runs/bc_pretrain_full_20260922_211200/split_manifest.json \
  --episode-id episode_000043 \
  --frame-skip 125
```

The checker first compares the reset FK position against HDF5
`obs/tcp_pose[t0]`, where `t0` is the first valid policy transition. If the
error exceeds 1 cm it stops before replay. It then
reports the first raw `executed_delta_joint` replay and the full contiguous
Expert policy segment, with 125 physics steps per action.
