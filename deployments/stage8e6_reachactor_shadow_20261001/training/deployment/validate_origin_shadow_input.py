#!/usr/bin/env python3
"""Validate an Origin read-only JSONL export before Actor shadow replay.

This tool is intentionally offline: it reads a file, performs schema checks and
never opens a socket, ROS node or RealMan connection. Missing measurements are
reported as blockers; they are never replaced with zeroes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REACH_REQUIRED = ('timestamp_s', 'joint_position_rad', 'tcp_pose', 'goal_pose',
                  'arm_err', 'sys_err', 'pose_origin_status')
SKILL_REQUIRED = ('phase', 'apple_position_m', 'apple_rotation',
                  'gripper_aperture_m', 'gripper_command',
                  'initial_apple_height_m', 'hold_fraction')


def _finite_shape(value, shape):
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return x.shape == shape and bool(np.all(np.isfinite(x)))


def _pose_ok(value):
    if not isinstance(value, dict): return False
    return (_finite_shape(value.get('position_m'), (3,)) and
            _finite_shape(value.get('rotation'), (3, 3)))


def validate_record(record, index):
    errors=[]; mode=record.get('mode')
    if mode not in ('reach','grasp_lift'): errors.append('mode:UNKNOWN')
    for key in REACH_REQUIRED:
        if key not in record or record[key] is None: errors.append(f'{key}:UNAVAILABLE')
    if record.get('pose_origin_status') != 'CONFIRMED_POLICY_TCP':
        errors.append('tcp_pose_origin:UNRESOLVED')
    for key in ('arm_err','sys_err'):
        if record.get(key) is not None and (isinstance(record[key], bool) or
                not isinstance(record[key], int)):
            errors.append(f'{key}:INVALID')
    if not isinstance(record.get('timestamp_s'), (int,float)) or not np.isfinite(record['timestamp_s']):
        errors.append('timestamp_s:INVALID')
    if not _finite_shape(record.get('joint_position_rad'), (6,)): errors.append('joint_position_rad:INVALID')
    for key in ('tcp_pose','goal_pose'):
        if key in record and record[key] is not None and not _pose_ok(record[key]): errors.append(f'{key}:INVALID')
    if mode == 'grasp_lift':
        for key in SKILL_REQUIRED:
            if key not in record or record[key] is None: errors.append(f'{key}:UNAVAILABLE')
        if record.get('apple_position_m') is not None and not _finite_shape(record['apple_position_m'],(3,)): errors.append('apple_position_m:INVALID')
        if record.get('apple_rotation') is not None and not _finite_shape(record['apple_rotation'],(3,3)): errors.append('apple_rotation:INVALID')
        if record.get('hold_fraction') is not None and (not np.isfinite(record['hold_fraction']) or not 0<=record['hold_fraction']<=1): errors.append('hold_fraction:INVALID')
    return dict(index=index, mode=mode, valid=not errors, errors=errors)


def validate_jsonl(path):
    records=[]; timestamps=[]
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip(): continue
        try:
            record=json.loads(line)
            check=validate_record(record,index)
            records.append(check)
            if isinstance(record.get('timestamp_s'), (int,float)) and np.isfinite(record['timestamp_s']): timestamps.append(float(record['timestamp_s']))
        except json.JSONDecodeError as exc:
            records.append(dict(index=index,mode=None,valid=False,errors=[f'json:{exc.msg}']))
    intervals=np.diff(np.asarray(timestamps)) if len(timestamps)>1 else np.empty(0)
    by_mode={m:[x for x in records if x['mode']==m] for m in ('reach','grasp_lift')}
    report=dict(records=len(records),valid=sum(x['valid'] for x in records),invalid=sum(not x['valid'] for x in records),
        reach=dict(records=len(by_mode['reach']),valid=sum(x['valid'] for x in by_mode['reach']),
                   policy_ready=bool(by_mode['reach']) and all(x['valid'] for x in by_mode['reach'])),
        grasp_lift=dict(records=len(by_mode['grasp_lift']),valid=sum(x['valid'] for x in by_mode['grasp_lift']),
                        policy_ready=bool(by_mode['grasp_lift']) and all(x['valid'] for x in by_mode['grasp_lift'])),
        interval_s=(dict(median=float(np.median(intervals)),p95=float(np.percentile(intervals,95)),max=float(intervals.max())) if len(intervals) else None),
        error_counts={})
    counts={}
    for row in records:
        for error in row['errors']: counts[error]=counts.get(error,0)+1
    report['error_counts']=counts
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('jsonl',nargs='?')
    p.add_argument('--output')
    a=p.parse_args()
    if not a.jsonl:
        print(json.dumps(dict(mode='offline-validator',network=False,hardware_sent=False,
                              reach_required=REACH_REQUIRED,skill_required=SKILL_REQUIRED),indent=2)); return
    report=validate_jsonl(a.jsonl)
    if a.output: Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
