#!/usr/bin/env python3
"""One supervised ReachActor step; waits for literal EXECUTE before any write."""
import json, math, socket, sys, time, statistics
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import String
import tf2_ros

ROOT=Path('/home/nvidia/RL-Orchard')
BUNDLE=ROOT/'deployments/stage8e6_reachactor_shadow_20261001'
RUN=BUNDLE/'training/runs/stage8b2_1_cartesian_bc_dagger_20260929'
OUT=Path('/tmp/stage8e12_live_single_step')
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(BUNDLE))
sys.path.insert(0,'/home/nvidia/RL-Orchard/src/orchard_picker/src')
from training.models import SACActor
from training.deployment.pose_math import quaternion_xyzw_to_matrix, matrix_to_rotvec, compose_pose
from orchard_picker.protocol import pose_to_protocol

HOST='192.168.1.18'; PORT=8080; TCP_TIMEOUT=3.0
TCP_OFFSET=np.asarray([0.0005,-0.005,0.158],dtype=np.float64)
LOW=np.asarray([-0.0015,-0.0015,-0.0015,-0.0075,-0.0075,-0.0075],dtype=np.float64)
HIGH=-LOW

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def matrix_to_quaternion_xyzw(R):
    t=float(np.trace(R))
    if t>0:
        s=math.sqrt(t+1.0)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
    q=np.asarray([x,y,z,w],dtype=np.float64); return (q/np.linalg.norm(q)).tolist()

def decode_udp(data, source, now):
    p=json.loads(data.decode('utf-8').strip()); js=p.get('joint_status'); wp=p.get('waypoint')
    if not isinstance(js,dict) or not isinstance(wp,dict): raise ValueError('missing fields')
    jp=js.get('joint_position'); pos=wp.get('position'); eu=wp.get('euler')
    if not (isinstance(jp,list) and len(jp)==6 and isinstance(pos,list) and len(pos)==3 and isinstance(eu,list) and len(eu)==3): raise ValueError('bad dimensions')
    return {'host_timestamp_s':now,'source_ip':source[0],'source_port':source[1],'raw_payload':p,'joint_raw':jp,'joint_rad':[float(x)*0.001*math.pi/180.0 for x in jp],'waypoint_position_raw':pos,'waypoint_euler_raw':eu,'waypoint_quaternion_raw':wp.get('quat',wp.get('quaternion')),'err':p.get('err'),'joint_err_code':js.get('joint_err_code'),'joint_en_flag':js.get('joint_en_flag')}

class Inputs(Node):
    def __init__(self):
        super().__init__('stage8e12_supervised_preflight')
        self.tfbuf=tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.listener=tf2_ros.TransformListener(self.tfbuf,self,spin_thread=False)
        self.target=None; self.info={}; self.udp=None; self.samples=[]; self.targets=[]; self.malformed=0
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); self.sock.bind(('192.168.1.100',8090)); self.sock.setblocking(False)
        self.create_subscription(PointStamped,'/apple/target_point',self.on_target,10)
        self.create_subscription(String,'/apple/target_info',self.on_info,10)
    def on_info(self,msg):
        try: self.info=json.loads(msg.data)
        except Exception: pass
    def on_target(self,msg):
        if msg.header.frame_id!='camera_link': return
        xyz=np.asarray([msg.point.x,msg.point.y,msg.point.z],dtype=np.float64)
        if np.all(np.isfinite(xyz)):
            self.target={'xyz':xyz,'frame':'camera_link','ros_timestamp_s':float(msg.header.stamp.sec)+float(msg.header.stamp.nanosec)*1e-9,'receive_s':time.monotonic()}
            self.targets.append({'host_timestamp_s':self.target['receive_s'],'ros_timestamp_s':self.target['ros_timestamp_s'],'apple_camera_xyz_m':xyz.tolist(),'frame':'camera_link','confidence':self.info.get('confidence'),'target_valid':self.info.get('target_valid')})
    def poll_udp(self):
        while True:
            try: data,src=self.sock.recvfrom(65535)
            except BlockingIOError: return
            except Exception: return
            try:
                row=decode_udp(data,src,time.monotonic()); self.udp=row; self.samples.append(row)
            except Exception: self.malformed+=1
    def tf(self,parent,child):
        try:
            t=self.tfbuf.lookup_transform(parent,child,Time()); tr=t.transform.translation; q=t.transform.rotation
            R=quaternion_xyzw_to_matrix([q.x,q.y,q.z,q.w]); p=np.asarray([tr.x,tr.y,tr.z],dtype=np.float64)
            return p,R,[q.x,q.y,q.z,q.w],t.header.stamp
        except Exception: return None
    def close(self): self.sock.close(); self.destroy_node()

def read_until_line(sock):
    buf=bytearray(); deadline=time.monotonic()+TCP_TIMEOUT
    while b'\n' not in buf:
        rem=deadline-time.monotonic()
        if rem<=0: raise TimeoutError('response timeout')
        sock.settimeout(rem); chunk=sock.recv(4096)
        if not chunk: break
        buf.extend(chunk)
    return bytes(buf).split(b'\n',1)[0].rstrip(b'\r').decode('utf-8')

def main():
    rclpy.init(); node=Inputs(); deadline=time.monotonic()+15.0; candidate=None; held=None
    cp=torch.load(RUN/'actor_best.pt',map_location='cpu',weights_only=False); meta={k:cp.get(k) for k in ('state_dim','action_dim','hidden_dim','architecture_version','action_definition','action_source','evaluation_policy')}
    actor=SACActor(15,6,int(meta['hidden_dim'])); actor.load_state_dict(cp['model_state_dict']); actor.eval()
    z=np.load(RUN/'normalization.npz'); mean=np.asarray(z['state_mean'],dtype=np.float64); std=np.asarray(z['state_std'],dtype=np.float64); action_low=np.asarray(z['action_low'],dtype=np.float64); action_high=np.asarray(z['action_high'],dtype=np.float64)
    while time.monotonic()<deadline and candidate is None:
        rclpy.spin_once(node,timeout_sec=0.01); node.poll_udp()
        if node.udp is None or node.target is None: continue
        if time.monotonic()-node.udp['host_timestamp_s']>0.15 or time.monotonic()-node.target['receive_s']>2.0: continue
        err=node.udp.get('err'); jerr=node.udp.get('joint_err_code') or []; en=node.udp.get('joint_en_flag') or []
        if any(int(x)!=0 for x in (err if isinstance(err,list) else [err])) or any(int(x)!=0 for x in jerr) or not all(bool(int(x)) for x in en): continue
        cam=node.tf('base_link','camera_link'); cur=node.tf('base_link','robot_end')
        if cam is None or cur is None: continue
        pbc,Rbc,_,_=cam; pbe,Rbe,qbe,stamp=cur
        if held is None: held=Rbe.copy()
        apple_base=pbc+Rbc@node.target['xyz']; tip_target=apple_base; target_end=tip_target-Rbe@TCP_OFFSET; rel=target_end-pbe; rerr=matrix_to_rotvec(held@Rbe.T)
        raw=np.r_[node.udp['joint_rad'],pbe,rel,rerr]
        if raw.shape!=(15,) or not np.all(np.isfinite(raw)): continue
        norm=(raw-mean)/std
        with torch.no_grad(): a=actor.deterministic_action(torch.from_numpy(norm.astype(np.float32)).unsqueeze(0))[0].numpy().astype(np.float64)
        raw_physical=a*action_high
        # Numerical endpoint rounding only: keep the actor output and enforce exact training bounds.
        # The frozen file contains tiny floating-point endpoint noise; the
        # deployment contract is exactly +/-0.0015 m and +/-0.0075 rad.
        physical=np.clip(raw_physical,LOW,HIGH)
        dot=float(np.dot(physical[:3],rel))
        if not np.all(np.isfinite(physical)) or dot<=0: continue
        if any(abs(physical[i])>abs(rel[i]) and physical[i]*rel[i]>0 for i in range(3)): continue
        next_p,next_R=compose_pose(pbe,Rbe,physical[:3],physical[3:]); qnext=matrix_to_quaternion_xyzw(next_R)
        pose=Pose(); pose.position.x,pose.position.y,pose.position.z=[float(x) for x in next_p]; pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=[float(x) for x in qnext]
        protocol_pose=pose_to_protocol(pose); payload={'command':'movep_canfd','pose':protocol_pose,'follow':False,'trajectory_mode':0,'radio':0}
        target_record={'xyz':node.target['xyz'].tolist(),'frame':node.target['frame'],'ros_timestamp_s':node.target['ros_timestamp_s'],'receive_s':node.target['receive_s']}
        candidate={'created_monotonic_s':time.monotonic(),'robot_feedback':node.udp,'target':target_record,'current_robot_end_position_m':pbe.tolist(),'current_robot_end_quaternion_xyzw':qbe,'target_robot_end_position_m':target_end.tolist(),'goal_relative_xyz':rel.tolist(),'held_orientation_source':'HELD_RUNTIME_EPISODE_START','raw_observation_15d':raw.tolist(),'normalized_action':a.tolist(),'raw_physical_action':raw_physical.tolist(),'physical_action_bounded':physical.tolist(),'numerical_bound_clipping':(not np.array_equal(raw_physical,physical)),'goal_dot_action':dot,'next_robot_end_position_m':next_p.tolist(),'next_robot_end_quaternion_xyzw':qnext,'protocol_pose':protocol_pose,'payload':payload,'tcp_offset_m':TCP_OFFSET.tolist(),'tcp_compensation_count':1,'controller_error':err,'joint_error':jerr,'joint_enable':en,'tf_stamp':{'sec':stamp.sec,'nanosec':stamp.nanosec}}
    if candidate is None:
        node.close(); rclpy.shutdown(); dump('preflight.json',{'status':'STAGE8E12_SINGLE_STEP_BLOCKED','reason':'fresh valid UDP/vision/TF/action gate not obtained','hardware_sent':False,'motion_command_count':0}); print('STAGE8E12_SINGLE_STEP_BLOCKED'); return 2
    dump('preflight.json',{'status':'READY_FOR_OPERATOR_GATE','checkpoint_metadata':meta,'state_dim':15,'action_dim':6,'udp_packet_count':len(node.samples),'udp_malformed':node.malformed,'vision_sample_count':len(node.targets),'latest_target_age_ms':(time.monotonic()-node.target['receive_s'])*1000.0,'candidate':candidate,'hardware_sent':False,'motion_command_count':0})
    print('\n=== FINAL HUMAN GATE ===')
    print('CURRENT TCP:',candidate['current_robot_end_position_m'])
    print('TARGET ROBOT_END:',candidate['target_robot_end_position_m'])
    print('GOAL RELATIVE:',candidate['goal_relative_xyz'])
    print('ACTOR STEP TRANSLATION (mm):',[x*1000.0 for x in candidate['physical_action_bounded'][:3]])
    print('ACTOR STEP ROTATION (rad):',candidate['physical_action_bounded'][3:])
    print('GOAL DOT ACTION:',candidate['goal_dot_action'])
    print('NEXT ROBOT_END:',candidate['next_robot_end_position_m'])
    print('PROTOCOL POSE:',candidate['protocol_pose'])
    print('TARGET AGE (ms):',(time.monotonic()-candidate['target']['receive_s'])*1000.0)
    print('Enter exactly EXECUTE to send one Movep_CANFD, or anything else to cancel.',flush=True)
    answer=sys.stdin.readline().strip()
    if answer!='EXECUTE':
        candidate['cancelled']=True; dump('command_record.json',{'status':'CANCELLED','answer':answer,'hardware_sent':False,'motion_command_count':0,'payload':candidate['payload']}); node.close(); rclpy.shutdown(); print('STAGE8E12_SINGLE_STEP_CANCELLED'); return 0
    # Re-check freshness at the exact gate.  A delayed approval never sends
    # a stale target or stale feedback; it must be re-preflighted instead.
    rclpy.spin_once(node, timeout_sec=0.02); node.poll_udp()
    if node.udp is None or time.monotonic()-node.udp['host_timestamp_s']>0.15 or time.monotonic()-node.target['receive_s']>2.0:
        dump('command_record.json',{'status':'BLOCKED_STALE_AT_EXECUTE_GATE','hardware_sent':False,'motion_command_count':0,'payload':candidate['payload']}); node.close(); rclpy.shutdown(); print('STAGE8E12_SINGLE_STEP_BLOCKED_STALE_AT_EXECUTE_GATE'); return 2
    # No command socket is opened before the explicit gate.
    sock=socket.create_connection((HOST,PORT),timeout=TCP_TIMEOUT); sock.settimeout(TCP_TIMEOUT)
    wire=(json.dumps(candidate['payload'],separators=(',',':'),ensure_ascii=False)+'\r\n').encode('utf-8'); sent=time.monotonic(); sock.sendall(wire); candidate['motion_command_count']=1
    response=None; response_error=None
    try: response=read_until_line(sock)
    except Exception as exc: response_error=repr(exc)
    sock.close(); dump('command_record.json',{'status':'SENT_ONCE','command_type':'Movep_CANFD','command_pose':candidate['protocol_pose'],'normalized_action':candidate['normalized_action'],'physical_action':candidate['physical_action_bounded'],'timestamp':sent,'follow':False,'trajectory_mode':0,'radio':0,'wire_utf8_repr':repr(wire),'response':response,'response_error':response_error,'hardware_sent':True,'motion_command_count':1})
    # Observe only UDP for three seconds; do not run Actor or send a second command.
    end=time.monotonic()+3.0; trajectory=[]
    while time.monotonic()<end:
        node.poll_udp()
        if node.udp is not None: trajectory.append(node.udp)
        time.sleep(0.005)
    dump('feedback_before.json',candidate['robot_feedback']); dump('feedback_trajectory.jsonl',''.join(json.dumps(x,separators=(',',':'))+'\n' for x in trajectory)); dump('safety_counters.json',{'hardware_sent':True,'command_writer':False,'motion_command_count':1,'Movep_CANFD_count':1,'MoveJ_count':0,'MoveL_count':0,'servo_count':0,'gripper_command_count':0,'home_count':0}); node.close(); rclpy.shutdown(); print('STAGE8E12_SINGLE_STEP_SENT_ONCE'); return 0

if __name__=='__main__': raise SystemExit(main())
