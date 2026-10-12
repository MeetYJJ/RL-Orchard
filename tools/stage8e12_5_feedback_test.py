#!/usr/bin/env python3
"""Stage8E12.5: supervised +X feedback-source motion test.

The only writable operation is the explicitly gated movep_canfd command.  All
state comparisons use the receive-only UDP stream and a read-only TCP
get_current_arm_state query.
"""
import json, math, socket, sys, time
import numpy as np

ROOT='/home/nvidia/RL-Orchard'; sys.path.insert(0,ROOT+'/src/orchard_picker/src')
from geometry_msgs.msg import Pose
from orchard_picker.protocol import protocol_to_pose, pose_to_protocol

HOST='192.168.1.18'; PORT=8080; LOCAL=('192.168.1.100',8090)
STEP_M=0.001; MAX_STEPS=20; WAIT_AFTER=0.40; TCP_TIMEOUT=0.20

def decode_udp(data,src):
    p=json.loads(data.decode('utf-8').strip()); js=p.get('joint_status'); wp=p.get('waypoint')
    if not isinstance(js,dict) or not isinstance(wp,dict): raise ValueError('missing UDP state fields')
    jp=js.get('joint_position'); pos=wp.get('position'); eu=wp.get('euler')
    if not (isinstance(jp,list) and len(jp)==6 and isinstance(pos,list) and len(pos)==3 and isinstance(eu,list) and len(eu)==3): raise ValueError('invalid UDP state dimensions')
    return {'receive_s':time.monotonic(),'raw':p,'position_m':np.asarray(pos,dtype=np.float64)/1e6,'joint_deg':np.asarray(jp,dtype=np.float64)/1000.0,'euler_raw':list(eu),'err':p.get('err'),'joint_err_code':js.get('joint_err_code'),'source':f'{src[0]}:{src[1]}'}

def valid_udp(u):
    if u is None or time.monotonic()-u['receive_s']>0.20:return False
    if not np.all(np.isfinite(u['position_m'])) or not np.all(np.isfinite(u['joint_deg'])):return False
    e=u['err'] if isinstance(u['err'],list) else [u['err']]
    return all(int(v)==0 for v in e) and all(int(v)==0 for v in (u['joint_err_code'] or []))

def tcp_readonly_arm_state():
    wire=b'{"command":"get_current_arm_state"}\r\n'; t0=time.monotonic(); s=socket.create_connection((HOST,PORT),timeout=2.0); s.settimeout(TCP_TIMEOUT); s.sendall(wire)
    buf=bytearray()
    try:
        while b'\n' not in buf:
            chunk=s.recv(8192)
            if not chunk: break
            buf.extend(chunk)
    finally: s.close()
    raw=bytes(buf).split(b'\n',1)[0].rstrip(b'\r').decode('utf-8'); msg=json.loads(raw); arm=msg.get('arm_state',msg); pose=arm.get('pose'); joint=arm.get('joint')
    if not (isinstance(pose,list) and len(pose)>=6 and isinstance(joint,list) and len(joint)>=6): raise ValueError('invalid get_current_arm_state response')
    err=arm.get('err',arm.get('arm_err',None));
    if isinstance(err,list): err=[int(v) for v in err]
    elif err is not None: err=[int(err)]
    return {'receive_s':time.monotonic(),'rtt_s':time.monotonic()-t0,'raw':msg,'position_m':np.asarray(pose[:3],dtype=np.float64)/1e6,'joint_deg':np.asarray(joint[:6],dtype=np.float64)/1000.0,'pose_raw':list(pose[:6]),'err':err}

def send_movep(protocol_pose):
    payload={'command':'movep_canfd','pose':[int(v) for v in protocol_pose],'follow':False,'trajectory_mode':0,'radio':0}; wire=(json.dumps(payload,separators=(',',':'))+'\r\n').encode('utf-8'); s=socket.create_connection((HOST,PORT),timeout=3.0); s.settimeout(0.05); s.sendall(wire); timed_out=False; response=None
    try:
        buf=bytearray()
        while b'\n' not in buf:
            chunk=s.recv(4096)
            if not chunk: break
            buf.extend(chunk)
        response=bytes(buf).split(b'\n',1)[0].rstrip(b'\r').decode('utf-8')
    except (socket.timeout,TimeoutError): timed_out=True
    finally:s.close()
    return payload,timed_out,response

def main():
    print('STAGE8E12.5 READY'); print('DIRECTION=base X positive'); print('STEP=+1.0 mm'); print('MAX_STEPS=20'); print('MAX_TOTAL=+20.0 mm'); print('Read-only TCP state query and UDP feedback will be sampled before the gate.');
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); sock.bind(LOCAL); sock.setblocking(False); latest=None; malformed=0
    deadline=time.monotonic()+3.0
    while time.monotonic()<deadline:
        try:
            data,src=sock.recvfrom(65535); latest=decode_udp(data,src)
        except BlockingIOError: time.sleep(0.005)
        except Exception: malformed+=1
    if not valid_udp(latest): print('STAGE8E12.5_BLOCKED_UDP_UNAVAILABLE'); sock.close(); return 0
    try: arm0=tcp_readonly_arm_state()
    except Exception as exc: print('STAGE8E12.5_BLOCKED_ARM_STATE_UNAVAILABLE',repr(exc)); sock.close(); return 0
    start_udp=latest['position_m'].copy(); start_arm=arm0['position_m'].copy(); start_joint=arm0['joint_deg'].copy(); print('START_UDP_XYZ =',np.round(start_udp,6).tolist()); print('START_JOINT =',np.round(latest['joint_deg'],4).tolist()); print('START_ARM_STATE_XYZ =',np.round(start_arm,6).tolist()); print('START_ARM_STATE_JOINT =',np.round(start_joint,4).tolist()); print('START_CONTROLLER_ERROR =',latest['err'],arm0['err']); print('Enter START20 to begin, or anything else to cancel.',flush=True)
    if sys.stdin.readline().strip()!='START20': print('STAGE8E12.5_CANCELLED'); sock.close(); return 0
    commands=0; timeout_count=0; last_udp=latest; last_arm=arm0; final_status='PHYSICAL_MOTION_NOT_CONFIRMED'; stopped_reason=''
    for step in range(1,MAX_STEPS+1):
        # Refresh the current waypoint immediately before constructing this step.
        wait=time.monotonic()+0.5
        while time.monotonic()<wait:
            try:
                data,src=sock.recvfrom(65535); u=decode_udp(data,src)
                if valid_udp(u): last_udp=u; break
            except BlockingIOError: time.sleep(0.003)
            except Exception: malformed+=1
        if not valid_udp(last_udp): stopped_reason='UDP_STALE'; break
        cur_pose=protocol_to_pose(list(last_udp['raw']['waypoint']['position'])+list(last_udp['raw']['waypoint']['euler'])); cur_pose.position.x += STEP_M; protocol_pose=pose_to_protocol(cur_pose)
        try: payload,timed_out,response=send_movep(protocol_pose)
        except Exception as exc: stopped_reason='TRANSPORT_ERROR:'+repr(exc); break
        commands+=1
        if timed_out: timeout_count+=1; print('TCP_TIMEOUT_NO_RETRY')
        end=time.monotonic()+WAIT_AFTER; after_udp=None
        while time.monotonic()<end:
            try:
                data,src=sock.recvfrom(65535); u=decode_udp(data,src)
                if valid_udp(u): after_udp=u
            except BlockingIOError: pass
            except Exception: malformed+=1
            time.sleep(0.002)
        if after_udp is None: stopped_reason='UDP_FEEDBACK_UNCONFIRMED'; break
        try: after_arm=tcp_readonly_arm_state()
        except Exception as exc: stopped_reason='ARM_STATE_QUERY_FAILED:'+repr(exc); break
        udp_delta=(after_udp['position_m']-start_udp)*1000.0; arm_delta=(after_arm['position_m']-start_arm)*1000.0; joint_delta=after_arm['joint_deg']-start_joint
        print('STEP %02d/%02d' % (step,MAX_STEPS)); print('command_total_x_mm = %.3f' % (step*1.0)); print('udp_xyz_delta_mm =',np.round(udp_delta,3).tolist()); print('arm_state_xyz_delta_mm =',np.round(arm_delta,3).tolist()); print('joint_delta_deg =',np.round(joint_delta,4).tolist()); print('udp_joint_current =',np.round(after_udp['joint_deg'],4).tolist()); print('controller_error =',after_udp['err'],'arm_state_error =',after_arm['err'],'joint_error =',after_udp['joint_err_code'])
        if (after_udp['err'] and any(int(v)!=0 for v in (after_udp['err'] if isinstance(after_udp['err'],list) else [after_udp['err']]))) or (after_arm['err'] and any(int(v)!=0 for v in after_arm['err'])) or any(int(v)!=0 for v in (after_udp['joint_err_code'] or [])): stopped_reason='ROBOT_ERROR'; break
        last_udp=after_udp; last_arm=after_arm
        if step>=5 and np.linalg.norm(udp_delta)<0.5 and np.linalg.norm(arm_delta)<0.5 and np.linalg.norm(joint_delta)<0.05:
            print('COMMANDS_SENT_BUT_PHYSICAL_MOTION_NOT_CONFIRMED'); stopped_reason='PHYSICAL_MOTION_NOT_CONFIRMED'; break
    udp_total=(last_udp['position_m']-start_udp)*1000.0; arm_total=(last_arm['position_m']-start_arm)*1000.0; joint_total=last_arm['joint_deg']-start_joint
    if not stopped_reason and commands==MAX_STEPS: final_status='PHYSICAL_MOTION_CONFIRMED'; stopped_reason='completed'
    elif np.linalg.norm(udp_total)>5.0 and np.linalg.norm(joint_total)>0.05 and np.linalg.norm(arm_total)>5.0: final_status='PHYSICAL_MOTION_CONFIRMED'; stopped_reason=stopped_reason or 'feedback_agrees'
    elif np.linalg.norm(udp_total)>5.0 and np.linalg.norm(joint_total)<0.05 and np.linalg.norm(arm_total)<0.5: final_status='UDP_WAYPOINT_NOT_VALID_AS_PHYSICAL_MOTION_FEEDBACK'
    elif np.linalg.norm(joint_total)>0.05 and np.linalg.norm(arm_total)>5.0 and np.linalg.norm(udp_total)<0.5: final_status='UDP_WAYPOINT_FEEDBACK_MISMATCH'
    print('commands_sent =',commands); print('commanded_total_x_mm =',commands*1.0); print('UDP: udp_waypoint_total_delta_mm =',np.round(udp_total,3).tolist()); print('ARM STATE: arm_state_total_delta_mm =',np.round(arm_total,3).tolist()); print('JOINT: joint_total_delta_deg =',np.round(joint_total,4).tolist()); print('tcp_readonly_timeout_count =',timeout_count); print('udp_malformed_count =',malformed); print('physical_motion_visual_confirmation = OPERATOR_REQUIRED'); print('final_status =',final_status); sock.close(); return 0

if __name__=='__main__': raise SystemExit(main())
