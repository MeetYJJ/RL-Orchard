#!/usr/bin/env python3
"""Stage8E14B: cross-validated eye-in-hand ReachActor finite horizon.

This runtime uses only the already verified Movep_CANFD JSON path.  Vision
state is separated from stable-target publication: a short target-point gap
can use the last *base-frame* target, but a stale camera-frame point is never
reused or re-transformed.
"""
import json, math, socket, sys, time
import numpy as np
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from apple_interfaces.msg import ApplePositionArray
import yaml

ROOT='/home/nvidia/RL-Orchard'
BUNDLE=ROOT+'/deployments/stage8e6_reachactor_shadow_20261001'
RUN=BUNDLE+'/training/runs/stage8b2_1_cartesian_bc_dagger_20260929'
sys.path.insert(0,BUNDLE); sys.path.insert(0,ROOT+'/src/orchard_picker/src')
from training.models import SACActor
from training.deployment.pose_math import quaternion_xyzw_to_matrix, matrix_to_rotvec, compose_pose
from training.deployment.skill_runtime_contract import REACH_ACTION_BOUNDS
from orchard_picker.protocol import pose_to_protocol, protocol_to_pose

HOST='192.168.1.18'; PORT=8080; MAX_STEPS=250; PERIOD=0.200
TARGET_UPDATE_MAX_AGE=0.300
FROZEN_TARGET_CONTROL_MAX_AGE=1.000
SIGNAL_ALIVE_MAX_AGE=0.500
TRUE_LOSS_CONFIRM=1.000
UDP_MAX_AGE=0.150; RESPONSE_TIMEOUT=0.050; OBSERVE_WINDOW=0.160
DIST_NOISE_TOL=0.0001; REACH_THRESHOLD=0.005
TCP_OFFSET=np.asarray([0.0005,-0.005,0.158],dtype=np.float64)
BOUNDS=np.asarray(REACH_ACTION_BOUNDS,dtype=np.float64)
HAND_EYE_CONFIG='/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml'

def load_hand_eye():
    with open(HAND_EYE_CONFIG,'r') as f:
        root=yaml.safe_load(f)
    p=root['hand_eye_static_tf_publisher']['ros__parameters']
    if p.get('enabled') is False or p.get('parent_frame')!='robot_end' or p.get('child_frame')!='camera_link' or bool(p.get('invert_matrix',False)):
        raise RuntimeError('hand-eye config is not the verified robot_end -> camera_link transform')
    R=np.asarray(p['rotation_matrix'],dtype=np.float64).reshape(3,3)
    t=np.asarray(p['translation_m'],dtype=np.float64)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise RuntimeError('invalid hand-eye matrix in picker_params.yaml')
    return R,t

def q_from_R(R):
    t=float(np.trace(R))
    if t>0:
        s=math.sqrt(t+1)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
    q=np.asarray([x,y,z,w],dtype=np.float64); return (q/np.linalg.norm(q)).tolist()

def decode(data,src):
    p=json.loads(data.decode().strip()); js=p.get('joint_status'); wp=p.get('waypoint')
    if not isinstance(js,dict) or not isinstance(wp,dict): raise ValueError('missing state fields')
    jp=js.get('joint_position'); pos=wp.get('position'); eu=wp.get('euler')
    if not (isinstance(jp,list) and len(jp)==6 and isinstance(pos,list) and len(pos)==3 and isinstance(eu,list) and len(eu)==3): raise ValueError('invalid state dimensions')
    pose=protocol_to_pose(list(pos)+list(eu)); q=[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
    return {'host_s':time.monotonic(),'source':f'{src[0]}:{src[1]}','raw':p,'joint_rad':[float(v)*0.001*math.pi/180 for v in jp], 'position_m':[float(v)/1e6 for v in pos], 'euler_raw':eu,'quaternion_xyzw':q,'err':p.get('err'),'joint_err_code':js.get('joint_err_code'),'joint_en_flag':js.get('joint_en_flag')}

class Inputs(Node):
    def __init__(self):
        super().__init__('stage8e14b_finite_reach')
        self.hand_eye_R,self.hand_eye_T=load_hand_eye()
        self.target=None; self.info={}; self.latest=None; self.malformed=0; self.sources=set()
        self.last_apples_receipt=None; self.apple_count=0; self.last_rgb_receipt=None; self.last_depth_receipt=None
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); self.sock.bind(('192.168.1.100',8090)); self.sock.setblocking(False)
        self.create_subscription(PointStamped,'/apple/target_point',self.on_target,10)
        self.create_subscription(String,'/apple/target_info',self.on_info,10)
        self.create_subscription(ApplePositionArray,'/apple_positions',self.on_apples,10)
        self.create_subscription(Image,'/apple_vision/rgb',self.on_rgb,10)
        self.create_subscription(Image,'/apple_vision/aligned_depth',self.on_depth,10)
    def on_info(self,msg):
        try:self.info=json.loads(msg.data)
        except Exception:pass
    def on_target(self,msg):
        if msg.header.frame_id!='camera_link': return
        xyz=np.asarray([msg.point.x,msg.point.y,msg.point.z],dtype=np.float64)
        if np.all(np.isfinite(xyz)): self.target={'xyz':xyz,'receive_s':time.monotonic(),'ros_s':float(msg.header.stamp.sec)+float(msg.header.stamp.nanosec)*1e-9,'frame':'camera_link'}
    def on_apples(self,msg):
        self.last_apples_receipt=time.monotonic(); self.apple_count=len(msg.apples)
    def on_rgb(self,msg): self.last_rgb_receipt=time.monotonic()
    def on_depth(self,msg): self.last_depth_receipt=time.monotonic()
    def poll(self):
        while True:
            try:data,src=self.sock.recvfrom(65535)
            except BlockingIOError:return
            except Exception:return
            try:self.latest=decode(data,src); self.sources.add(self.latest['source'])
            except Exception:self.malformed+=1
    def camera_from_udp(self,pcur,Rcur):
        return pcur + Rcur @ self.hand_eye_T, Rcur @ self.hand_eye_R
    def close(self):self.sock.close(); self.destroy_node()

def valid_state(u):
    if u is None or time.monotonic()-u['host_s']>UDP_MAX_AGE:return False
    if not np.all(np.isfinite(u['joint_rad']+u['position_m'])):return False
    e=u['err'] if isinstance(u['err'],list) else [u['err']]
    return all(int(v)==0 for v in e) and all(int(v)==0 for v in (u['joint_err_code'] or [])) and all(bool(int(v)) for v in (u['joint_en_flag'] or []))

def send_once(payload):
    wire=(json.dumps(payload,separators=(',',':'))+'\r\n').encode('utf-8'); sock=socket.create_connection((HOST,PORT),timeout=3.0); sock.settimeout(RESPONSE_TIMEOUT); sent=time.monotonic(); sock.sendall(wire); response=None; timed_out=False
    try:
        buf=bytearray()
        while b'\n' not in buf:
            chunk=sock.recv(4096)
            if not chunk: break
            buf.extend(chunk)
        response=bytes(buf).split(b'\n',1)[0].rstrip(b'\r').decode('utf-8')
    except (socket.timeout,TimeoutError): timed_out=True
    finally:sock.close()
    return sent,response,timed_out

def vision_state(node, now, frozen_receipt):
    target_age=float('inf') if node.target is None else now-node.target['receive_s']
    apples_alive=node.last_apples_receipt is not None and now-node.last_apples_receipt<=SIGNAL_ALIVE_MAX_AGE
    camera_alive=(node.last_rgb_receipt is not None and now-node.last_rgb_receipt<=SIGNAL_ALIVE_MAX_AGE and node.last_depth_receipt is not None and now-node.last_depth_receipt<=SIGNAL_ALIVE_MAX_AGE)
    detection_alive=apples_alive and node.apple_count>0
    if node.target is not None and target_age<=TARGET_UPDATE_MAX_AGE:
        return 'TRACKING',target_age,0.0,apples_alive,camera_alive,detection_alive
    frozen_age=float('inf') if frozen_receipt is None else now-frozen_receipt
    if apples_alive and detection_alive and camera_alive:
        if frozen_receipt is not None and frozen_age<=FROZEN_TARGET_CONTROL_MAX_AGE:
            return 'TEMP',target_age,frozen_age,apples_alive,camera_alive,detection_alive
        return 'TEMP_WAIT',target_age,frozen_age,apples_alive,camera_alive,detection_alive
    return 'LOST_CANDIDATE',target_age,frozen_age,apples_alive,camera_alive,detection_alive

def main():
    print('STAGE8E14B READY'); print('MAX_STEPS = 250'); print('CONTROL_RATE = 5 Hz'); print('TARGET_UPDATE_MAX_AGE = 300 ms'); print('FROZEN_TARGET_CONTROL_MAX_AGE = 1000 ms'); print('MAX_TRANSLATION_PER_AXIS = 1.5 mm'); print('MAX_ROTATION_PER_AXIS = 0.0075 rad'); print('Enter START250 to begin, or anything else to cancel.',flush=True)
    if sys.stdin.readline().strip()!='START250': print('STAGE8E14B_ROBOT_ERROR'); return 0
    rclpy.init(); node=Inputs(); print('HAND_EYE_SOURCE=%s' % HAND_EYE_CONFIG,flush=True); held=None; frozen_target=None; frozen_receipt=None; state_prev=None; failure_since=None; had_target=False; command_count=0; initial_distance=None; final_distance=None; sat_axes=0; total_axes=0; temp_cycles=0; temp_wait_cycles=0; tracking_cycles=0; true_loss_count=0; reacq_count=0; timeout_count=0; stale_count=0; status='STAGE8E14B_MAX_STEPS_REACHED'; final_err=None; final_jerr=None
    cp=torch.load(RUN+'/actor_best.pt',map_location='cpu',weights_only=False); actor=SACActor(15,6,int(cp['hidden_dim'])); actor.load_state_dict(cp['model_state_dict']); actor.eval(); z=np.load(RUN+'/normalization.npz'); mean=np.asarray(z['state_mean'],dtype=np.float64); std=np.asarray(z['state_std'],dtype=np.float64)
    try:
        for step in range(1,MAX_STEPS+1):
            cycle_start=time.monotonic(); candidate=None; acquisition_window=10.0 if step==1 else 0.18
            while time.monotonic()-cycle_start<acquisition_window:
                rclpy.spin_once(node,timeout_sec=0.005); node.poll(); now=time.monotonic()
                if not valid_state(node.latest): continue
                vstate,tage,fage,alive,cam_alive,det_alive=vision_state(node,now,frozen_receipt)
                if vstate=='LOST_CANDIDATE':
                    if not had_target:
                        continue
                    if failure_since is None: failure_since=now
                    if now-failure_since>=TRUE_LOSS_CONFIRM:
                        if state_prev!='LOST': print('VISION: %s -> LOST | reason=detector_or_camera_loss_confirmed' % (state_prev or 'UNKNOWN'))
                        true_loss_count+=1; status='STAGE8E14B_TRUE_VISION_LOSS'; break
                    continue
                failure_since=None
                if state_prev!=vstate:
                    reason='stable_target_gap' if vstate in ('TEMP','TEMP_WAIT') else ('stable_target_reacquired' if state_prev in ('TEMP','TEMP_WAIT') else 'fresh_target')
                    print('VISION: %s -> %s | reason=%s' % (state_prev or 'UNKNOWN',vstate,reason));
                    if state_prev in ('TEMP','TEMP_WAIT') and vstate=='TRACKING': reacq_count+=1
                    state_prev=vstate
                if vstate=='TEMP_WAIT': temp_wait_cycles+=1; continue
                if vstate=='TEMP': temp_cycles+=1
                else: tracking_cycles+=1
                u=node.latest; pose=protocol_to_pose(list(np.asarray(u['position_m'])*1e6)+list(u['euler_raw'])); q=[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]; Rcur=quaternion_xyzw_to_matrix(q); pcur=np.asarray(u['position_m']); pbc,Rbc=node.camera_from_udp(pcur,Rcur)
                if held is None: held=Rcur.copy(); print('HELD_RUNTIME_EPISODE_START')
                if vstate=='TRACKING':
                    had_target=True; apple_base=pbc+Rbc@node.target['xyz']; frozen_target=apple_base-Rcur@TCP_OFFSET; frozen_receipt=node.target['receive_s']; frozen_age=0.0
                if frozen_target is None: continue
                target_end=frozen_target.copy(); goal=target_end-pcur; rerr=matrix_to_rotvec(held@Rcur.T); raw=np.r_[u['joint_rad'],pcur,goal,rerr]
                if raw.shape!=(15,) or not np.all(np.isfinite(raw)): continue
                norm=(raw-mean)/std
                with torch.no_grad(): an=actor.deterministic_action(torch.from_numpy(norm.astype(np.float32)).unsqueeze(0))[0].numpy().astype(np.float64)
                if not np.all(np.isfinite(an)): continue
                ap=np.clip(an,-1.0,1.0)*BOUNDS; sat=np.abs(np.abs(an)-1.0)<1e-7
                for i in range(3):
                    if ap[i]*goal[i]>0 and abs(ap[i])>abs(goal[i]): ap[i]=goal[i]
                dot=float(np.dot(goal,ap[:3])); dist=float(np.linalg.norm(goal))
                if dot<=0: status='STAGE8E14B_MOTION_VALIDATION_FAILED'; break
                if initial_distance is None: initial_distance=dist
                if dist<=REACH_THRESHOLD: status='STAGE8E14B_TARGET_REACHED'; final_distance=dist; break
                nxt_p,nxt_R=compose_pose(pcur,Rcur,ap[:3],ap[3:]); nq=q_from_R(nxt_R); pp=Pose(); pp.position.x,pp.position.y,pp.position.z=[float(v) for v in nxt_p]; pp.orientation.x,pp.orientation.y,pp.orientation.z,pp.orientation.w=nq; protocol_pose=pose_to_protocol(pp); payload={'command':'movep_canfd','pose':protocol_pose,'follow':False,'trajectory_mode':0,'radio':0}; candidate=(u,target_end.copy(),goal.copy(),ap.copy(),sat,pcur.copy(),dist,payload)
                break
            if status in ('STAGE8E14B_TRUE_VISION_LOSS','STAGE8E14B_TARGET_REACHED','STAGE8E14B_MOTION_VALIDATION_FAILED'):
                break
            if candidate is None:
                if status=='STAGE8E14B_TRUE_VISION_LOSS': break
                vstate,tage,fage,alive,cam_alive,det_alive=vision_state(node,time.monotonic(),frozen_receipt)
                if vstate=='TEMP_WAIT': print('WAITING_FOR_STABLE_TARGET_REACQUISITION')
                elif vstate=='LOST_CANDIDATE': stale_count+=1
                continue
            u,target_before,goal,ap,sat,pbefore,dist_before,payload=candidate; sat_axes+=int(np.sum(sat[:3])); total_axes+=3
            try: sent,response,timed_out=send_once(payload)
            except Exception: status='STAGE8E14B_MOTION_VALIDATION_FAILED'; break
            command_count+=1
            if timed_out: timeout_count+=1; print('TCP_RESPONSE_TIMEOUT_NO_RETRY')
            observe_end=time.monotonic()+OBSERVE_WINDOW; after=None
            while time.monotonic()<observe_end:
                rclpy.spin_once(node,timeout_sec=0.002); node.poll()
                if valid_state(node.latest): after=node.latest
            if after is None: status='STAGE8E14B_MOTION_VALIDATION_FAILED'; break
            actual=np.asarray(after['position_m'])-pbefore; req=ap[:3]; goal_dot_actual=float(np.dot(goal,actual)); req_dot_actual=float(np.dot(req,actual)); dist_after=float(np.linalg.norm(target_before-np.asarray(after['position_m']))); final_distance=dist_after; final_err=after['err']; final_jerr=after['joint_err_code']; vstate,tage,fage,_,_,_=vision_state(node,time.monotonic(),frozen_receipt)
            if req_dot_actual<=0 or goal_dot_actual<=0 or dist_after>dist_before+DIST_NOISE_TOL: status='STAGE8E14B_MOTION_VALIDATION_FAILED'; break
            print('STEP %03d/%03d | vision=%s | target_age=%.0fms | frozen_age=%.0fms | dist=%.1f->%.1fmm | act=%smm | actual=%smm | sat=%s | err=0' % (step,MAX_STEPS,vstate,tage*1000,fage*1000,dist_before*1000,dist_after*1000,np.round(req*1000,3).tolist(),np.round(actual*1000,3).tolist(),bool(np.any(sat[:3]))))
            if dist_after<=REACH_THRESHOLD: status='STAGE8E14B_TARGET_REACHED'; break
            time.sleep(max(0.0,PERIOD-(time.monotonic()-cycle_start)))
    finally:
        if final_err is None and node.latest is not None: final_err=node.latest['err']
        if final_jerr is None and node.latest is not None: final_jerr=node.latest['joint_err_code']
        node.close(); rclpy.shutdown()
    print('STAGE8E14B RESULT'); print('final_status = %s' % status); print('steps_executed = %d' % command_count); print('motion_command_count = %d' % command_count); print('initial_distance_mm = %s' % ('%.3f'%(initial_distance*1000) if initial_distance is not None else 'N/A')); print('final_distance_mm = %s' % ('%.3f'%(final_distance*1000) if final_distance is not None else 'N/A')); print('distance_reduction_mm = %s' % ('%.3f'%((initial_distance-final_distance)*1000) if initial_distance is not None and final_distance is not None else 'N/A')); print('tracking_cycles = %d' % tracking_cycles); print('temporary_unqualified_cycles = %d' % temp_cycles); print('temp_wait_cycles = %d' % temp_wait_cycles); print('true_loss_count = %d' % true_loss_count); print('stable_target_reacquisition_count = %d' % reacq_count); print('max_frozen_target_age_ms = %.1f' % (FROZEN_TARGET_CONTROL_MAX_AGE*1000)); print('translation_saturation_ratio = %s' % ('%.3f'%(sat_axes/total_axes) if total_axes else 'N/A')); print('controller_error = %s' % final_err); print('joint_error = %s' % final_jerr)
    return 0

if __name__=='__main__': raise SystemExit(main())
