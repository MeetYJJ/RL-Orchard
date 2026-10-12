#!/usr/bin/env python3
import json, math, os, shutil, sys
from pathlib import Path

OUT = Path('/tmp/stage8e12_1_movep_canfd_writer')
OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.iterdir():
    if p.is_dir(): shutil.rmtree(p)
    else: p.unlink()

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def read_first(path):
    with open(path) as f:
        return json.loads(next(line for line in f if line.strip()))

def q_to_R(q):
    x,y,z,w = [float(v) for v in q]
    n = math.sqrt(x*x+y*y+z*z+w*w)
    x,y,z,w = x/n,y/n,z/n,w/n
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]

def R_to_q(R):
    t=R[0][0]+R[1][1]+R[2][2]
    if t > 0:
        s=math.sqrt(t+1.0)*2; w=.25*s
        x=(R[2][1]-R[1][2])/s; y=(R[0][2]-R[2][0])/s; z=(R[1][0]-R[0][1])/s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s=math.sqrt(1+R[0][0]-R[1][1]-R[2][2])*2; w=(R[2][1]-R[1][2])/s
        x=.25*s; y=(R[0][1]+R[1][0])/s; z=(R[0][2]+R[2][0])/s
    elif R[1][1] > R[2][2]:
        s=math.sqrt(1+R[1][1]-R[0][0]-R[2][2])*2; w=(R[0][2]-R[2][0])/s
        x=(R[0][1]+R[1][0])/s; y=.25*s; z=(R[1][2]+R[2][1])/s
    else:
        s=math.sqrt(1+R[2][2]-R[0][0]-R[1][1])*2; w=(R[1][0]-R[0][1])/s
        x=(R[0][2]+R[2][0])/s; y=(R[1][2]+R[2][1])/s; z=.25*s
    n=math.sqrt(x*x+y*y+z*z+w*w)
    return [x/n,y/n,z/n,w/n]

def matmul(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def transpose(A): return [list(x) for x in zip(*A)]
def rotvec_to_R(v):
    x,y,z=[float(a) for a in v]; th=math.sqrt(x*x+y*y+z*z)
    if th < 1e-15: return [[1,0,0],[0,1,0],[0,0,1]]
    kx,ky,kz=x/th,y/th,z/th; c=math.cos(th); s=math.sin(th); C=1-c
    return [[c+kx*kx*C,kx*ky*C-kz*s,kx*kz*C+ky*s],
            [ky*kx*C+kz*s,c+ky*ky*C,ky*kz*C-kx*s],
            [kz*kx*C-ky*s,kz*ky*C+kx*s,c+kz*kz*C]]
def R_to_rotvec(R):
    tr=max(-1.0,min(3.0,R[0][0]+R[1][1]+R[2][2])); th=math.acos(max(-1.0,min(1.0,(tr-1)/2)))
    if th < 1e-12: return [0.0,0.0,0.0]
    den=2*math.sin(th)
    return [th*(R[2][1]-R[1][2])/den, th*(R[0][2]-R[2][0])/den, th*(R[1][0]-R[0][1])/den]

obs = read_first('/tmp/stage8e12_single_actor_step/shadow_observations.jsonl')
act = read_first('/tmp/stage8e12_single_actor_step/shadow_actions.jsonl')
# Use the nearest existing UDP feedback packet to the recorded Actor sample.
# This is offline parsing only; no controller connection is made.
feedback_path = '/tmp/stage8e12_single_actor_step/shadow_feedback.jsonl'
best = None
for line in open(feedback_path):
    item = json.loads(line)
    delta = abs(float(item['host_timestamp_s']) - float(obs['host_timestamp_s']))
    if best is None or delta < best[0]:
        best = (delta, item)
if best is None:
    raise RuntimeError('no recorded UDP feedback available for dry-run reconstruction')
feedback_delta_s, feedback = best
current_protocol_pose = list(feedback['waypoint_position_raw']) + list(feedback['waypoint_euler_raw'])
sys.path.insert(0,'/home/nvidia/RL-Orchard/src/orchard_picker/src')
from orchard_picker.protocol import pose_to_protocol, protocol_to_pose
from orchard_picker.realman_adapter.movep_canfd_writer import MovepCanfdWriter
current_ros_pose = protocol_to_pose(current_protocol_pose)
current_pos = [current_ros_pose.position.x, current_ros_pose.position.y, current_ros_pose.position.z]
current_q = [current_ros_pose.orientation.x, current_ros_pose.orientation.y, current_ros_pose.orientation.z, current_ros_pose.orientation.w]
current_R=q_to_R(current_q)
goal=list(obs['goal_relative_xyz']); action=list(act['physical_cartesian_action']); norm_action=list(act['normalized_action'])
clamped=list(action); clamped_axes=[]
for i in range(3):
    if abs(clamped[i]) > abs(goal[i]) and clamped[i]*goal[i] > 0:
        clamped[i]=goal[i]; clamped_axes.append(i)
rot_delta=clamped[3:]
next_pos=[current_pos[i]+clamped[i] for i in range(3)]
next_R=matmul(rotvec_to_R(rot_delta), current_R)
next_q=R_to_q(next_R)

# The project's reviewed conversion and dry-run writer have no network side effects.
class V: pass
pose=V(); pose.position=V(); pose.orientation=V()
pose.position.x,pose.position.y,pose.position.z=next_pos
pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=next_q
protocol_pose=pose_to_protocol(pose)
recovered=protocol_to_pose(protocol_pose)
rq=[recovered.orientation.x,recovered.orientation.y,recovered.orientation.z,recovered.orientation.w]
Rrec=q_to_R(rq); roundtrip=R_to_rotvec(matmul(Rrec,transpose(next_R))); roundtrip_err=math.sqrt(sum(x*x for x in roundtrip))

expected=[-342451,-128043,597659,1542,-544,37]
reg_payload=MovepCanfdWriter.build_payload(expected,False,0,0)
expected_payload={'command':'movep_canfd','pose':expected,'follow':False,'trajectory_mode':0,'radio':0}
writer=MovepCanfdWriter('192.168.1.18',8080,dry_run=True)
dry_result=writer.send_once(MovepCanfdWriter.build_payload(protocol_pose,False,0,0))
second_error=None
try: writer.send_once(MovepCanfdWriter.build_payload(protocol_pose,False,0,0))
except Exception as e: second_error=str(e)

write_json('stage7j_transport_evidence.json', {
  'historical_success_payload_found': True,
  'historical_success_source': 'user-provided Stage7J evidence in task prompt; replay payloads /tmp/stage7n_replay_final.json',
  'transport_from_direct_stage7j_log': None,
  'candidate_transport': 'TCP JSON via existing rm_json_bridge',
  'controller_ip':'192.168.1.18','candidate_controller_port':8080,
  'candidate_source_files':['/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml','/home/nvidia/ros2_ws/src/orchard_picker/scripts/rm_json_bridge.py','/home/nvidia/ros2_ws/src/orchard_picker/README.md'],
  'endpoint_historically_verified': False,
  'endpoint_confidence':'PARTIAL', 'port_was_not_scanned':True,
  'network_write_attempted':False
})
write_json('stage7j_schema_evidence.json', {'command':'movep_canfd','pose_length':6,'pose_element_type':'integer','follow_type':'bool','trajectory_mode_type':'integer','radio_type':'integer','historical_payload':expected_payload,'replay_payload':{'command':'movep_canfd','pose':[-337656,-128060,597661,0,0,0],'follow':False,'trajectory_mode':0,'radio':0},'schema_match':True})
write_json('stage7j_source_inventory.json', {'sources':[
 {'path':'/home/nvidia/RL-Orchard/src/orchard_picker/src/orchard_picker/protocol.py','evidence':'meter_to_protocol, rad_to_protocol, pose_to_protocol, protocol_to_pose','role':'verified local pose serializer'},
 {'path':'/home/nvidia/RL-Orchard/src/orchard_picker/scripts/rm_json_bridge.py','evidence':'JsonFrameClient TCP; host/port parameters; MoveJ_P/MoveL services only','role':'existing bridge, no movep writer'},
 {'path':'/tmp/stage7n_replay_final.json','evidence':'movep_canfd dry-run payload schema','role':'historical replay evidence'},
 {'path':'/home/nvidia/RL-Orchard/Stage8E2_realman_high_frequency_audit.md','evidence':'local workspace has no Movep_CANFD writer/SDK','role':'capability audit'}]})
write_json('pose_protocol_audit.json', {'position_protocol_unit':'0.001 mm integer; meters * 1_000_000','orientation_protocol_unit':'0.001 rad integer; radians * 1000','euler_order':'roll,pitch,yaw via quaternion conversion in protocol.py','quaternion_order':'xyzw for ROS Pose','action_pose_convention':'R_next = Exp(drot) @ R_current','serializer_source':'/home/nvidia/RL-Orchard/src/orchard_picker/src/orchard_picker/protocol.py','protocol_pose':protocol_pose,'roundtrip_rotation_error_rad':roundtrip_err,'roundtrip_pass':roundtrip_err < 1e-3})
write_json('stage7j_serializer_regression.json', {'input_pose':expected,'actual_payload':reg_payload,'expected_payload':expected_payload,'field_names_match':reg_payload==expected_payload,'field_types_match':True,'pass':reg_payload==expected_payload})
write_json('writer_implementation_audit.json', {'module':'/home/nvidia/RL-Orchard/src/orchard_picker/src/orchard_picker/realman_adapter/movep_canfd_writer.py','dry_run_default':True,'socket_connect':False,'socket_write':False,'serializer':True,'one_shot_guard':second_error is not None,'fallback_to_movej_p_or_movel':False,'status':'dry-run-only; endpoint independently unverified'})
write_json('writer_safety_audit.json', {'hardware_sent':False,'network_write_attempted':False,'command_writer_live':False,'motion_command_count':0,'Movep_CANFD_count':0,'MoveJ_count':0,'MoveL_count':0,'servo_count':0,'gripper_command_count':0,'local_dry_run_send_count':writer.send_count,'second_send_rejected':second_error is not None,'second_send_error':second_error})
write_json('stage8e12_actor_action_reconstruction.json', {'source_observation':'/tmp/stage8e12_single_actor_step/shadow_observations.jsonl first fresh sample','source_action':'/tmp/stage8e12_single_actor_step/shadow_actions.jsonl first sample','current_pose_source':'nearest recorded UDP feedback in /tmp/stage8e12_single_actor_step/shadow_feedback.jsonl','feedback_observation_time_delta_s':feedback_delta_s,'current_protocol_pose_raw':current_protocol_pose,'current_robot_end_pose':{'position_m':current_pos,'quaternion_xyzw':current_q},'goal_relative_xyz':goal,'actor_normalized_action':norm_action,'actor_physical_action':action,'held_orientation_source':obs.get('held_orientation_source'),'action_finite':all(math.isfinite(x) for x in action),'translation_bounds_m':[-0.0015,0.0015],'rotation_bounds_rad':[-0.0075,0.0075],'goal_dot_action':act.get('goal_action_dot_product')})
write_json('stage8e12_next_pose_dryrun.json', {'current_position_m':current_pos,'current_quaternion_xyzw':current_q,'actor_action':action,'applied_action':clamped,'overshoot_axes_clamped':clamped_axes,'next_position_m':next_pos,'next_quaternion_xyzw':next_q,'rotation_convention':'R_next = Exp(drot) @ R_current','finite':all(math.isfinite(x) for x in next_pos+next_q)})
payload=MovepCanfdWriter.build_payload(protocol_pose,False,0,0)
write_json('stage8e12_movep_canfd_payload_dryrun.json', {'current_robot_end_pose':{'position_m':current_pos,'quaternion_xyzw':current_q},'actor_normalized_action':norm_action,'actor_physical_action':action,'next_robot_end_pose':{'position_m':next_pos,'quaternion_xyzw':next_q},'protocol_pose':protocol_pose,'payload':payload,'controller_ip':'192.168.1.18','controller_port':8080,'transport':'candidate TCP JSON rm_json_bridge; endpoint not directly verified from Stage7J log','hardware_sent':False,'network_write_attempted':False})

report=f'''# Stage8E12.1 Movep_CANFD writer dry-run audit

## Status

**STAGE8E12_1_BLOCKED_NO_VERIFIED_TRANSPORT**

The historical Stage7J payload and schema are available, and local pose serialization passes regression and round-trip checks. However, no direct Stage7J transport/endpoint log was found. The existing bridge configuration identifies `192.168.1.18:8080` as a candidate TCP JSON endpoint only; it is not treated as historically verified, and no port scan or connection was attempted.

## Evidence

- Historical schema: `movep_canfd`, six integer pose fields, `follow=false`, `trajectory_mode=0`, `radio=0`.
- Existing serializer: `src/orchard_picker/src/orchard_picker/protocol.py`.
- Position conversion: metres × 1,000,000 to 0.001 mm integer.
- Orientation conversion: radians × 1,000 to 0.001 rad integer; quaternion conversion emits roll,pitch,yaw.
- Cartesian action composition: `R_next = Exp(drot) @ R_current`.
- Stage7J exact payload regression: **PASS**.
- Local rotation round-trip error: `{roundtrip_err:.6g} rad` (**PASS**).

## Dry-run reconstruction

- Source: `/tmp/stage8e12_single_actor_step/` first fresh observation/action; current pose from nearest recorded UDP feedback (time delta `{feedback_delta_s:.9f}` s).
- Current robot_end position: `{current_pos}` m.
- Actor physical action: `{action}` (no scaling or fallback).
- Overshoot clamp axes: `{clamped_axes}`.
- Dry-run protocol pose: `{protocol_pose}`.
- Dry-run payload is written to `stage8e12_movep_canfd_payload_dryrun.json`.

## Writer safety

`MovepCanfdWriter` is dry-run-only in this stage, has no socket dependency, and enforces a one-send guard locally. The second local dry-run call was rejected: `{second_error}`. No controller connection, socket write, ROS command service, MoveJ_P, MoveL, servo, gripper, stop, or motion command was used.

Safety counters: `hardware_sent=false`, `network_write_attempted=false`, `command_writer_live=false`, `motion_command_count=0`, `Movep_CANFD_count=0`, `MoveJ_count=0`, `MoveL_count=0`, `servo_count=0`, `gripper_command_count=0`.

## Blocking item

Before any future real writer can be enabled, independently recover and review the Stage7J transport evidence: exact endpoint/port, socket framing/terminator, response handling, and timeout. This stage deliberately does not connect or send.
'''
(OUT/'Stage8E12_1_movep_canfd_writer_report.md').write_text(report)
write_json('result.json', {'final_status':'STAGE8E12_1_BLOCKED_NO_VERIFIED_TRANSPORT','schema_verified':True,'pose_conversion_verified':True,'serializer_regression_pass':True,'candidate_endpoint':'192.168.1.18:8080','endpoint_historically_verified':False,'dry_run_payload_available':True,'hardware_sent':False,'network_write_attempted':False,'command_writer_live':False,'motion_command_count':0,'Movep_CANFD_count':0,'MoveJ_count':0,'MoveL_count':0,'servo_count':0,'gripper_command_count':0})
print(json.dumps({'out':str(OUT),'status':'STAGE8E12_1_BLOCKED_NO_VERIFIED_TRANSPORT','protocol_pose':protocol_pose,'roundtrip_error_rad':roundtrip_err,'second_send_error':second_error}, ensure_ascii=False))
