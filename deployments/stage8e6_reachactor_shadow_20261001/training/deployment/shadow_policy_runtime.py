#!/usr/bin/env python3
"""Offline Origin feedback -> frozen Actors -> MockCommandWriter runner.

Input is an explicit JSONL feedback export. This file never imports ROS,
RealMan, MuJoCo or socket and never creates a hardware connection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models import SACActor
try:
    from .cartesian_controller_core import Pose
    from .pose_math import validate_rotation_matrix
    from .skill_runtime_contract import (
        ActionSafetyGate, HardwareFeedback, MockCommandWriter, ObservationBuilder,
        RuntimeStateMachine, normalized_reach_action, normalized_skill_action,
    )
    from .validate_origin_shadow_input import validate_record
except ImportError:  # direct ``python training/deployment/shadow_policy_runtime.py``
    from training.deployment.cartesian_controller_core import Pose
    from training.deployment.pose_math import validate_rotation_matrix
    from training.deployment.skill_runtime_contract import (
        ActionSafetyGate, HardwareFeedback, MockCommandWriter, ObservationBuilder,
        RuntimeStateMachine, normalized_reach_action, normalized_skill_action,
    )
    from training.deployment.validate_origin_shadow_input import validate_record


REACH_RUN = ROOT/'training/runs/stage8b2_1_cartesian_bc_dagger_20260929'
SKILL_RUN = ROOT/'training/runs/stage8c2_2_grasp_lift_bc_dagger_20260930'


def _pose(obj, name):
    try:
        return Pose(np.asarray(obj['position_m'], dtype=float),
                    validate_rotation_matrix(np.asarray(obj['rotation'], dtype=float), name+' rotation'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'missing/invalid {name} pose: {exc}') from exc


class ShadowPolicyRuntime:
    def __init__(self, reach_run=REACH_RUN, skill_run=SKILL_RUN):
        self.reach_run=Path(reach_run); self.skill_run=Path(skill_run)
        rcp=torch.load(self.reach_run/'actor_best.pt',map_location='cpu',weights_only=False)
        scp=torch.load(self.skill_run/'actor_best.pt',map_location='cpu',weights_only=False)
        self.reach_actor=SACActor(15,6,256); self.reach_actor.load_state_dict(rcp['model_state_dict']); self.reach_actor.eval()
        self.skill_actor=SACActor(31,7,256); self.skill_actor.load_state_dict(scp['model_state_dict']); self.skill_actor.eval()
        rn=np.load(self.reach_run/'normalization.npz'); sn=np.load(self.skill_run/'normalization.npz')
        self.builder=ObservationBuilder(rn['state_mean'],rn['state_std'],sn['state_mean'],sn['state_std'])
        self.gate=ActionSafetyGate(); self.writer=MockCommandWriter(); self.state_machine=RuntimeStateMachine()
        self.state_machine.start()

    def process(self, record):
        validation = validate_record(record, 0)
        if not validation['valid']:
            raise ValueError('shadow input rejected: ' + ', '.join(validation['errors']))
        start=time.perf_counter(); mode=record.get('mode')
        if mode not in ('reach','grasp_lift'): raise ValueError('mode must be reach or grasp_lift')
        feedback=HardwareFeedback(
            np.asarray(record['joint_position_rad'],dtype=float),
            _pose(record['tcp_pose'],'tcp'), float(record['timestamp_s']),
            int(record['arm_err']), int(record['sys_err']),
            None if mode=='reach' else np.asarray(record['apple_position_m'],dtype=float),
            None if mode=='reach' else validate_rotation_matrix(np.asarray(record['apple_rotation'],dtype=float),'apple rotation'),
            None if mode=='reach' else float(record['gripper_aperture_m']))
        goal=_pose(record['goal_pose'],'goal')
        obs_start=time.perf_counter()
        if mode=='reach':
            raw=self.builder._raw_reach(feedback,goal); normalized=((raw-self.builder.reach_mean)/self.builder.reach_std).astype(np.float32)
            with torch.no_grad(): action_norm=self.reach_actor.deterministic_action(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
            physical=normalized_reach_action(action_norm)
        else:
            phase=str(record['phase']); command=float(record['gripper_command'])
            normalized=self.builder.grasp_lift(feedback,goal,gripper_command=command,phase=phase,
                initial_apple_height_m=float(record['initial_apple_height_m']),hold_fraction=float(record['hold_fraction']))
            raw=normalized*self.builder.skill_std+self.builder.skill_mean
            with torch.no_grad(): action_norm=self.skill_actor.deterministic_action(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
            physical=normalized_skill_action(action_norm)
        obs_ms=(time.perf_counter()-obs_start)*1000
        decision=self.gate.cartesian(physical[:6],feedback,feedback.timestamp_s)
        if not decision.accepted:
            log=dict(accepted=False,mode=mode,timestamp_s=feedback.timestamp_s,safety_reason=decision.reason,
                     observation=raw.tolist(),normalized_observation=normalized.tolist(),normalized_action=action_norm.tolist(),hardware_sent=False)
        else:
            self.writer.write_cartesian(decision.applied_action_m_rad,feedback.timestamp_s,source='policy')
            if mode=='grasp_lift': self.writer.write_gripper(float(physical[6]),feedback.timestamp_s,source='policy')
            log=dict(accepted=True,mode=mode,timestamp_s=feedback.timestamp_s,safety_reason=None,
                     observation=raw.tolist(),normalized_observation=normalized.tolist(),normalized_action=action_norm.tolist(),
                     physical_action=physical.tolist(),applied_action=decision.applied_action_m_rad.tolist(),hardware_sent=False)
        log.update(observation_latency_ms=obs_ms,inference_latency_ms=(time.perf_counter()-start)*1000-obs_ms,
                   phase=record.get('phase', 'REACH'),
                   feedback_age_not_measured=True)
        return log


def run_jsonl(input_path, output_path):
    runtime=ShadowPolicyRuntime(); logs=[]; timestamps=[]
    for line in Path(input_path).read_text().splitlines():
        if not line.strip(): continue
        try: logs.append(runtime.process(json.loads(line))); timestamps.append(float(json.loads(line)['timestamp_s']))
        except Exception as exc: logs.append({'accepted':False,'error':str(exc),'hardware_sent':False})
    ts=np.asarray(timestamps,dtype=float); intervals=np.diff(ts) if len(ts)>1 else np.empty(0)
    report=dict(records=len(logs),accepted=sum(x.get('accepted',False) for x in logs),
                rejected=sum(not x.get('accepted',False) for x in logs),
                feedback_interval_s=(dict(mean=float(intervals.mean()),p95=float(np.percentile(intervals,95)),max=float(intervals.max())) if len(intervals) else None),
                inference_latency_ms=(dict(mean=float(np.mean([x['inference_latency_ms'] for x in logs if 'inference_latency_ms' in x])),
                                           p95=float(np.percentile([x['inference_latency_ms'] for x in logs if 'inference_latency_ms' in x],95))) if any('inference_latency_ms' in x for x in logs) else None),
                hardware_sent=False,logs=logs)
    Path(output_path).write_text(json.dumps(report,indent=2)); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--feedback-jsonl'); p.add_argument('--output',default='shadow_policy_log.json')
    a=p.parse_args()
    if not a.feedback_jsonl:
        print(json.dumps(dict(mode='dry-run',network=False,hardware_sent=False,
            required_fields=['timestamp_s','joint_position_rad','tcp_pose','goal_pose'],
            skill_extra_fields=['phase','apple_position_m','apple_rotation','gripper_aperture_m','gripper_command','initial_apple_height_m','hold_fraction']),indent=2)); return
    print(json.dumps(run_jsonl(a.feedback_jsonl,a.output),indent=2))
if __name__=='__main__': main()
