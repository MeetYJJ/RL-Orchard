# BC pretraining for the future SAC actor

This directory is a read-only consumer of the existing `apple_rl_v1.2_sac_goal`
HDF5 episodes. It does not import ROS, start a process, or control a robot.
The first stage proves the complete data-to-network path:

```
HDF5 -> validated policy transitions -> normalized state/action
     -> SACActor mean branch -> BC loss -> checkpoint
```

## State and action

For transition `t`, the 12-dimensional state is

```
[obs/joint_pos[t][0:6],
 obs/tcp_pose[t][0:3],
 pregrasp_pose_base[0:3] - obs/tcp_pose[t][0:3]]
```

The six-dimensional expert action is
`actions/executed_delta_joint[t]` in radians. The loader verifies it against
`obs/joint_pos[t+1] - obs/joint_pos[t]`, and also verifies both `next_obs`
arrays. The frozen goal is read from `episode_meta/pregrasp_pose_base`, with a
frozen `teacher/pregrasp_pose_base` fallback. It is never recomputed from an
apple position.

Only episodes with `dataset_schema_version=apple_rl_v1.2_sac_goal`,
`episode_valid=true`, and `discarded!=true` are used. A transition must have
both `valid/policy_action` and `valid/executed_delta_joint` true. Thus scripted
grasp, gripper, retreat, and home transitions are excluded by the recorded
policy mask.

## Normalization

States use `(state - state_mean) / (state_std + 1e-8)`. Each action dimension
uses a scale equal to the maximum absolute expert delta in the selected
training subset. The resulting target is `delta_q / action_scale` and is
expected to be in `[-1, 1]`; any overflow is reported. This scale is only a
dataset/network normalization, **not** a real-robot safety limit or a joint
step limit.

## Actor and BC

`SACActor` is the future actor, not a separate BC model:

```
state[12] -> Linear(256) -> ReLU -> Linear(256) -> ReLU
          -> mean[6], log_std[6]
```

The BC stage trains the shared trunk and `mean_head` with
`MSE(tanh(mean), normalized_expert_action)`. `log_std_head` is present for
future SAC but frozen during BC. The saved `actor.pt` can therefore be loaded
as the actor when the later SAC critic and stochastic loss are added.

## Training-host migration

The training directory is deliberately independent of ROS. A training host
does not need ROS 2 Foxy, RealSense SDK, RealMan SDK, `rm_json_bridge`, the
GUI, or the ROS overlay. It needs Python, PyTorch, NumPy, and h5py. Python
3.10 is a practical recommendation; the code is also compatible with the
validated Python 3.8 syntax. Install PyTorch separately using the official
command selected for the training host GPU and NVIDIA driver. Do not copy the
Jetson-specific PyTorch build from the robot host.

The minimal non-CUDA dependency list is
`training/requirements-base.txt`:

```bash
conda create -n rl_orchard_train python=3.10 -y
conda activate rl_orchard_train
python -m pip install -r ~/RL-Orchard/training/requirements-base.txt
# Install torch using the command generated for this host's CPU/CUDA setup.
```

## Commands

Use the project Python/ROS environment if desired; the scripts themselves do
not start ROS:

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python /home/nvidia/RL-Orchard/training/inspect_training_dataset.py \
  --dataset-dir /home/nvidia/RL-Orchard/rl_dataset

python /home/nvidia/RL-Orchard/training/train_actor_bc_overfit.py \
  --dataset-dir /home/nvidia/RL-Orchard/rl_dataset \
  --max-transitions 32
```

The formal episode-level BC entry point is:

```bash
python /home/nvidia/RL-Orchard/training/train_actor_bc_full.py \
  --dataset-dir /home/nvidia/RL-Orchard/rl_dataset \
  --device auto --seed 42 --epochs 1000 --batch-size 64 \
  --learning-rate 0.001 --patience 50
```

It creates a train/validation/test split by episode, fits normalization on
train transitions only, selects `actor_best.pt` by validation MSE, and evaluates
the untouched test set once at the end. An existing `--split-manifest` can be
reused for fair comparison between models. The current repository does not yet
contain an online replay buffer or MuJoCo environment.

Each run writes `training/runs/bc_pretrain_overfit_*/` containing:

* `actor.pt` — SACActor checkpoint and architecture metadata;
* `normalization.npz` — state mean/std and action scales;
* `training_metrics.json`, `loss.csv`, and `config.yaml`.

`--max-transitions` intentionally selects a tiny set. This is an overfit
diagnostic, not a generalization benchmark. A later SAC implementation should
load `actor.pt` and the same normalization file; no second actor architecture
should be introduced.

## Stage 3A: offline transition and reward audit

Stage 3A is an audit/data layer only. It materializes validated
`(state, action, reward, next_state, done)` transitions and never updates a
critic, actor, replay buffer, or HDF5 file. The policy-subtask terminal is the
last valid policy transition in each episode; `terminal_success` is the
recorded `reached_pregrasp` flag. Whole-task `success` and `terminal_state`
remain diagnostics and do not discard a policy-successful episode.

The reward is a configurable provisional candidate, not a final SAC reward:

```
r = progress_weight * progress_m
    - action_weight * sum((delta_q / action_scale)**2)
    + success_bonus       (only on done & reached_pregrasp)
    - failure_penalty     (only on done & not reached_pregrasp)
```

Use the exact BC episode manifest and normalization file so Offline RL uses
the same episode benchmark and train-only statistics:

```bash
python training/inspect_offline_rl_dataset.py \
  --dataset-dir /path/to/apple_rl_dataset_v1_20260922 \
  --split-manifest training/runs/<bc-run>/split_manifest.json \
  --normalization training/runs/<bc-run>/normalization.npz \
  --actor-checkpoint training/runs/<bc-run>/actor_best.pt \
  --reward-config training/configs/offline_rl_reward_v0.yaml \
  --split all
```

The audit writes `offline_rl_audit.json`, `reward_statistics.json`,
`terminal_audit.json`, `problematic_episodes.json`, and `config.json` under an
`offline_rl_audit_*` directory. A DataLoader smoke batch checks
`state[B,12]`, `action[B,6]`, `reward[B,1]`, `next_state[B,12]`, and
`done[B,1]`. This stage does not implement SAC, Q networks, replay training,
ROS, MuJoCo, or robot control.

## Stage 3B: Twin-Q critic smoke test

Stage 3B adds only critic-side numerical sanity checks. `SACQNetwork` is an
independent MLP over `[state_norm, action_norm]` with dimensions
`18 -> 256 -> 256 -> 1`; two independent online critics and two frozen target
critics are created. The target is:

```
r + gamma * (1-done) * (min(target_q1, target_q2) - alpha * log_pi)
```

The actor is the existing BC-pretrained `SACActor`, used only under
`torch.no_grad()` for target actions. There is no actor optimizer, SAC actor
loss, replay training, or robot control in this stage. The smoke script first
supports `frozen_target` mode and then `dynamic_bellman` mode:

```bash
python training/train_critic_overfit.py \
  --dataset-dir /path/to/apple_rl_dataset_v1_20260922 \
  --split-manifest /path/to/bc_run/split_manifest.json \
  --normalization /path/to/bc_run/normalization.npz \
  --actor-checkpoint /path/to/bc_run/actor_best.pt \
  --reward-config training/configs/offline_rl_reward_v0.yaml \
  --config training/configs/critic_overfit.yaml \
  --mode both --device auto --max-transitions 64
```

Outputs are smoke-only checkpoints under `critic_overfit_*/frozen_target/`
and `critic_overfit_*/dynamic_bellman/`; they are not final SAC or deployment
checkpoints.

## Stage 3C: Offline SAC + BC

`train_offline_sac_bc.py` initializes the `SACActor` from the selected BC
checkpoint, unfreezes its `log_std`, and jointly trains twin critics and the
actor on the manifest's Train transitions. Validation evaluates Bellman MSE,
BC and physical action errors, action saturation, and `Q_policy` versus
`Q_data`; it does not make optimizer updates. The Test split is not loaded.
The settings in `offline_sac_bc_v0.yaml` are provisional.

Run a 64-transition, 10-epoch pipeline smoke first:

```bash
python training/train_offline_sac_bc.py \
  --dataset-dir /path/to/apple_rl_dataset_v1_20260922 \
  --split-manifest /path/to/bc_run/split_manifest.json \
  --normalization /path/to/bc_run/normalization.npz \
  --actor-checkpoint /path/to/bc_run/actor_best.pt \
  --config training/configs/offline_sac_bc_v0.yaml \
  --max-transitions 64 --epochs 10 --device auto
```

After the smoke passes, omit `--max-transitions` and select the full configured
epoch count. `bc_weight`, `alpha`, learning rates, safety thresholds, and
patience can be overridden on the command line. Runs save best and last Actor,
online Q, and target Q weights, plus the exact normalization and split manifest
copies, config, metrics, and history.
