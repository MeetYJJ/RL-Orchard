#!/usr/bin/env python3
"""Stage8E13: at most 30 supervised ReachActor steps at 5 Hz."""
import json, math, socket, sys, time
import numpy as np
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import String
import tf2_ros

ROOT='/home/nvidia/RL-Orchard'; BUNDLE=ROOT+'/deployments/stage8e6_reachactor_shadow_20261001'; RUN=BUNDLE+'/training/runs/stage8b2_1_cartesian_bc_dagger_20260929'
sys.path.insert(0,BUNDLE); sys.path.insert(0,ROOT+'/src/orchard_picker/src')
from training.models import SACActor
from training.deployment.pose_math import quaternion_xyzw_to_matrix, matrix_to_rotvec, compose_pose
from training.deployment.skill_runtime_contract import REACH_ACTION_BOUNDS
from orchard_picker.protocol import pose_to_protocol, protocol_to_pose

HOST='192.168.1.18'; PORT=8080; MAX_STEPS=30; PERIOD=0.200; TARGET_MAX_AGE=0.5; UDP_MAX_AGE=0.15; RESPONSE_TIMEOUT=0.05; OBSERVE_WINDOW=0.16; DIST_NOISE_TOL=0.0001; REACH_THRESHOLD=0.005
TCP_OFFSET=np.asarray([0.0005,-0.005,0.158],dtype=np.float64); BOUNDS=np.asarray(REACH_ACTION_BOUNDS,dtype=np.float64)

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
    return {'host_s':time.monotonic(),'source':f'{src[0]}:{src[1]}','raw':p,'joint_rad':[float(v)*0.001*math.pi/180 for v in jp],'position_m':[float(v)/1e6 for v in pos],'euler_raw':eu,'quaternion_xyzw':q,'err':p.get('err'),'joint_err_code':js.get('joint_err_code'),'joint_en_flag':js.get('joint_en_flag')}

class Inputs(Node):
    def __init__(self):
        super().__init__('stage8e13_finite_reach_readonly')
        self.tfbuf=tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0)); self.listener=tf2_ros.TransformListener(self.tfbuf,self,spin_thread=False)
        self.target=None; self.info={}; self.latest=None; self.malformed=0; self.sources=set()
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); self.sock.bind(('192.168.1.100',8090)); self.sock.setblocking(False)
        self.create_subscription(PointStamped,'/apple/target_point',self.on_target,10); self.create_subscription(String,'/apple/target_info',self.on_info,10)
    def on_info(self,msg):
        try:self.info=json.loads(msg.data)
        except Exception:pass
    def on_target(self,msg):
        if msg.header.frame_id!='camera_link': return
        xyz=np.asarray([msg.point.x,msg.point.y,msg.point.z],dtype=np.float64)
        if np.all(np.isfinite(xyz)): self.target={'xyz':xyz,'receive_s':time.monotonic(),'ros_s':float(msg.header.stamp.sec)+float(msg.header.stamp.nanosec)*1e-9,'frame':'camera_link'}
    def poll(self):
        while True:
            try:data,src=self.sock.recvfrom(65535)
            except BlockingIOError:return
            except Exception:return
            try:
                self.latest=decode(data,src); self.sources.add(self.latest['source'])
            except Exception:self.malformed+=1
    def tf(self,parent,child):
        try:
            t=self.tfbuf.lookup_transform(parent,child,Time()); tr=t.transform.translation; q=t.transform.rotation; R=quaternion_xyzw_to_matrix([q.x,q.y,q.z,q.w]); return np.asarray([tr.x,tr.y,tr.z],dtype=np.float64),R
        except Exception:return None
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

def main():
    print('STAGE8E13 READY'); print('MAX_STEPS = 30'); print('CONTROL_RATE = 5 Hz'); print('MAX_TRANSLATION_PER_AXIS = 1.5 mm'); print('MAX_ROTATION_PER_AXIS = 0.0075 rad'); print('Enter START30 to begin, or anything else to cancel.',flush=True)
    if sys.stdin.readline().strip()!='START30': print('STAGE8E13_STOPPED_TRANSPORT_ERROR'); return 0
    rclpy.init(); node=Inputs(); held=None; command_count=0; initial_distance=None; final_distance=None; sat_axes=0; total_axes=0; stale_count=0; timeout_count=0; status='STAGE8E13_30STEP_PASSED'; final_err=None; final_jerr=None
    cp=torch.load(RUN+'/actor_best.pt',map_location='cpu',weights_only=False); actor=SACActor(15,6,int(cp['hidden_dim'])); actor.load_state_dict(cp['model_state_dict']); actor.eval(); z=np.load(RUN+'/normalization.npz'); mean=np.asarray(z['state_mean'],dtype=np.float64); std=np.asarray(z['state_std'],dtype=np.float64)
    try:
        for step in range(1,MAX_STEPS+1):
            cycle_start=time.monotonic(); candidate=None
            acquisition_window = 0.6 if step == 1 else 0.18
            while time.monotonic()-cycle_start<acquisition_window:
                rclpy.spin_once(node,timeout_sec=0.005); node.poll()
                if valid_state(node.latest) and node.target is not None and time.monotonic()-node.target['receive_s']<=TARGET_MAX_AGE:
                    cam=node.tf('base_link','camera_link')
                    if cam is not None:
                        u=node.latest; pbc,Rbc=cam; pose=protocol_to_pose(list(np.asarray(u['position_m'])*1e6)+list(u['euler_raw'])); q=[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]; Rcur=quaternion_xyzw_to_matrix(q); pcur=np.asarray(u['position_m']);
                        if held is None: held=Rcur.copy()
                        apple_base=pbc+Rbc@node.target['xyz']; target_end=apple_base-Rcur@TCP_OFFSET; goal=target_end-pcur; rerr=matrix_to_rotvec(held@Rcur.T); raw=np.r_[u['joint_rad'],pcur,goal,rerr]
                        if raw.shape==(15,) and np.all(np.isfinite(raw)):
                            norm=(raw-mean)/std
                            with torch.no_grad(): an=actor.deterministic_action(torch.from_numpy(norm.astype(np.float32)).unsqueeze(0))[0].numpy().astype(np.float64)
                            if np.all(np.isfinite(an)):
                                ap=np.clip(an,-1.0,1.0)*BOUNDS; sat=np.abs(np.abs(an)-1.0)<1e-7
                                for i in range(3):
                                    if ap[i]*goal[i]>0 and abs(ap[i])>abs(goal[i]): ap[i]=goal[i]
                                dot=float(np.dot(goal,ap[:3])); dist=float(np.linalg.norm(goal))
                                if dot<=0: status='STAGE8E13_STOPPED_ACTION_DIRECTION'; break
                                if initial_distance is None: initial_distance=dist
                                if dist<=REACH_THRESHOLD: status='STAGE8E13_TARGET_REACHED_EARLY'; final_distance=dist; break
                                nxt_p,nxt_R=compose_pose(pcur,Rcur,ap[:3],ap[3:]); nq=q_from_R(nxt_R); pp=Pose(); pp.position.x,pp.position.y,pp.position.z=[float(v) for v in nxt_p]; pp.orientation.x,pp.orientation.y,pp.orientation.z,pp.orientation.w=nq; protocol_pose=pose_to_protocol(pp); payload={'command':'movep_canfd','pose':protocol_pose,'follow':False,'trajectory_mode':0,'radio':0}; candidate=(u, target_end.copy(), goal.copy(), ap.copy(), sat, pcur.copy(), dist, payload, raw.copy(), norm.copy()); break
            if status.startswith('STAGE8E13_STOPPED') or status=='STAGE8E13_TARGET_REACHED_EARLY': break
            if candidate is None:
                if node.target is None or time.monotonic()-node.target['receive_s']>TARGET_MAX_AGE: stale_count+=1; status='STAGE8E13_STOPPED_VISION_STALE'
                elif not valid_state(node.latest): status='STAGE8E13_STOPPED_ROBOT_ERROR'
                else: status='STAGE8E13_STOPPED_MOTION_MISMATCH'
                break
            u,target_before,goal,ap,sat,pbefore,dist_before,payload,raw,norm=candidate; sat_axes+=int(np.sum(sat[:3])); total_axes+=3
            try: sent,response,timed_out=send_once(payload)
            except Exception: status='STAGE8E13_STOPPED_TRANSPORT_ERROR'; break
            command_count+=1
            if timed_out: timeout_count+=1; print('TCP_RESPONSE_TIMEOUT_NO_RETRY')
            observe_end=time.monotonic()+OBSERVE_WINDOW; after=None
            while time.monotonic()<observe_end:
                rclpy.spin_once(node,timeout_sec=0.002); node.poll()
                if valid_state(node.latest): after=node.latest
            if after is None: status='STAGE8E13_STOPPED_MOTION_MISMATCH'; break
            actual=np.asarray(after['position_m'])-pbefore; req=ap[:3]; goal_dot_actual=float(np.dot(goal,actual)); req_dot_actual=float(np.dot(req,actual)); dist_after=float(np.linalg.norm(target_before-np.asarray(after['position_m']))); final_distance=dist_after; final_err=after['err']; final_jerr=after['joint_err_code']
            if req_dot_actual<=0 or goal_dot_actual<=0: status='STAGE8E13_STOPPED_MOTION_MISMATCH'; break
            if dist_after>dist_before+DIST_NOISE_TOL: status='STAGE8E13_STOPPED_DISTANCE_INCREASE'; break
            print('STEP %02d/%02d | goal_mm=%s | act_mm=%s | actual_mm=%s | dist=%.1f->%.1f mm | dot_req_actual=%.6g | dot_goal_actual=%.6g | sat=%s | vision_age=%.0fms | err=0' % (step,MAX_STEPS,np.round(goal*1000,3).tolist(),np.round(req*1000,3).tolist(),np.round(actual*1000,3).tolist(),dist_before*1000,dist_after*1000,req_dot_actual,goal_dot_actual,bool(np.any(sat[:3])),(time.monotonic()-node.target['receive_s'])*1000))
            if dist_after<=REACH_THRESHOLD: status='STAGE8E13_TARGET_REACHED_EARLY'; break
            wait=max(0.0,PERIOD-(time.monotonic()-cycle_start)); time.sleep(wait)
    finally:
        if final_err is None and node.latest is not None: final_err=node.latest['err']
        if final_jerr is None and node.latest is not None: final_jerr=node.latest['joint_err_code']
        node.close(); rclpy.shutdown()
    print('STAGE8E13 RESULT'); print('steps_executed = %d' % command_count); print('motion_command_count = %d' % command_count); print('initial_distance_mm = %s' % ('%.3f'%(initial_distance*1000) if initial_distance is not None else 'N/A')); print('final_distance_mm = %s' % ('%.3f'%(final_distance*1000) if final_distance is not None else 'N/A')); print('distance_reduction_mm = %s' % ('%.3f'%((initial_distance-final_distance)*1000) if initial_distance is not None and final_distance is not None else 'N/A')); print('translation_saturation_ratio = %s' % ('%.3f'%(sat_axes/total_axes) if total_axes else 'N/A')); print('vision_stale_count = %d' % stale_count); print('tcp_response_timeout_count = %d' % timeout_count); print('controller_error = %s' % final_err); print('joint_error = %s' % final_jerr); print('final_status = %s' % status)
    return 0

if __name__=='__main__': raise SystemExit(main())
