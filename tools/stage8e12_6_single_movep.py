#!/usr/bin/env python3
"""One authorized +10 mm RealMan Movep_CANFD feedback comparison."""
import json, math, socket, time
import numpy as np

HOST='192.168.1.18'; PORT=8080; LOCAL=('192.168.1.100',8090)
def recv_json_line(sock, timeout):
    sock.settimeout(timeout); buf=bytearray()
    while b'\n' not in buf:
        chunk=sock.recv(8192)
        if not chunk: break
        buf.extend(chunk)
    return json.loads(bytes(buf).split(b'\n',1)[0].rstrip(b'\r').decode('utf-8'))
def get_arm_state():
    s=socket.create_connection((HOST,PORT),timeout=2.0)
    try:
        s.sendall(b'{"command":"get_current_arm_state"}\r\n')
        msg=recv_json_line(s,1.0)
    finally:s.close()
    arm=msg.get('arm_state',msg); pose=arm.get('pose'); joint=arm.get('joint')
    if not isinstance(pose,list) or len(pose)<6 or not isinstance(joint,list) or len(joint)<6: raise RuntimeError('malformed get_current_arm_state response')
    return {'raw':list(pose[:6]),'joint_raw':list(joint[:6]),'err':arm.get('err',arm.get('arm_err')),'response':msg}
def udp_state(sock, duration):
    end=time.monotonic()+duration; latest=None; malformed=0
    while time.monotonic()<end:
        try:data,src=sock.recvfrom(65535)
        except BlockingIOError: time.sleep(.002); continue
        try:
            obj=json.loads(data.decode('utf-8').strip()); js=obj['joint_status']; wp=obj['waypoint']; pos=wp['position']; eu=wp['euler']; joint=js['joint_position']
            if len(pos)!=3 or len(eu)!=3 or len(joint)!=6: raise ValueError('invalid dimensions')
            latest={'raw':[int(v) for v in pos]+[int(v) for v in eu], 'joint_raw':[int(v) for v in joint], 'err':obj.get('err'), 'joint_err_code':js.get('joint_err_code'), 'receive_s':time.monotonic(), 'source':src}
        except Exception: malformed+=1
    return latest,malformed
def xyz_m(raw): return np.asarray(raw[:3],dtype=np.float64)/1e6
def joint_deg(raw): return np.asarray(raw,dtype=np.float64)/1000.0
def errors_zero(*values):
    for value in values:
        if value is None: continue
        vals=value if isinstance(value,(list,tuple)) else [value]
        if any(int(v)!=0 for v in vals): return False
    return True
def main():
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); sock.bind(LOCAL); sock.setblocking(False)
    before_udp,mal1=udp_state(sock,0.4)
    if before_udp is None or time.monotonic()-before_udp['receive_s']>0.2: raise SystemExit('PRECHECK_UDP_UNAVAILABLE')
    before_arm=get_arm_state()
    print('UDP_TCP_BEFORE_RAW =',before_udp['raw'],flush=True)
    print('ARM_STATE_TCP_BEFORE_RAW =',before_arm['raw'],flush=True)
    print('JOINT_BEFORE_RAW =',before_udp['joint_raw'],flush=True)
    print('UDP_TCP_BEFORE_M =',np.round(xyz_m(before_udp['raw']),9).tolist(),flush=True)
    print('ARM_STATE_TCP_BEFORE_M =',np.round(xyz_m(before_arm['raw']),9).tolist(),flush=True)
    print('JOINT_BEFORE_DEG =',np.round(joint_deg(before_udp['joint_raw']),6).tolist(),flush=True)
    diff=(xyz_m(before_udp['raw'])-xyz_m(before_arm['raw']))*1000.0
    print('BEFORE_SOURCE_DIFF_MM =',np.round(diff,6).tolist(),flush=True)
    print('BEFORE_ERRORS =',before_udp['err'],before_udp['joint_err_code'],before_arm['err'],flush=True)
    if np.max(np.abs(diff))>0.5:
        print('final_status = PRECHECK_TCP_SOURCE_MISMATCH'); sock.close(); return 0
    if not errors_zero(before_udp['err'],before_udp['joint_err_code'],before_arm['err']):
        print('final_status = PRECHECK_ROBOT_ERROR'); sock.close(); return 0
    if time.monotonic()-before_udp['receive_s']>0.2:
        print('final_status = PRECHECK_UDP_STALE'); sock.close(); return 0
    # Reuse the verified protocol conversion; target XYZ comes from the
    # independent TCP arm-state sample, while orientation is preserved.
    from geometry_msgs.msg import Pose
    import sys
    sys.path.insert(0,'/home/nvidia/RL-Orchard/src/orchard_picker/src')
    from orchard_picker.protocol import protocol_to_pose,pose_to_protocol
    current=protocol_to_pose(before_arm['raw']); current.position.x += 0.010
    command_pose=pose_to_protocol(current)
    payload={'command':'movep_canfd','pose':command_pose,'follow':False,'trajectory_mode':0,'radio':0}
    target_m=np.asarray([command_pose[0],command_pose[1],command_pose[2]],dtype=np.float64)/1e6
    print('CURRENT_TCP_M =',np.round(xyz_m(before_arm['raw']),9).tolist(),flush=True)
    print('TARGET_TCP_M =',np.round(target_m,9).tolist(),flush=True)
    print('EXPECTED_DELTA_MM = [10.0, 0.0, 0.0]',flush=True)
    print('MOVEP_PAYLOAD =',json.dumps(payload,separators=(',',':')),flush=True)
    # This invocation follows the user's explicit authorization to move without
    # an additional confirmation prompt. Exactly one sendall is permitted.
    s=socket.create_connection((HOST,PORT),timeout=3.0); s.settimeout(0.15)
    wire=(json.dumps(payload,separators=(',',':'))+'\r\n').encode('utf-8')
    s.sendall(wire); command_count=1; response_timeout=False
    try: recv_json_line(s,0.15)
    except (socket.timeout,TimeoutError): response_timeout=True; print('TCP_RESPONSE_TIMEOUT_NO_RETRY',flush=True)
    except Exception: response_timeout=True; print('TCP_RESPONSE_TIMEOUT_NO_RETRY',flush=True)
    finally:s.close()
    time.sleep(2.5)
    after_udp,mal2=udp_state(sock,0.25)
    after_arm=get_arm_state()
    if after_udp is None: raise SystemExit('AFTER_UDP_UNAVAILABLE command_count=1')
    udp_delta=(xyz_m(after_udp['raw'])-xyz_m(before_udp['raw']))*1000.0
    arm_delta=(xyz_m(after_arm['raw'])-xyz_m(before_arm['raw']))*1000.0
    joint_delta=joint_deg(after_udp['joint_raw'])-joint_deg(before_udp['joint_raw'])
    print('UDP_TCP_AFTER_RAW =',after_udp['raw'],flush=True)
    print('ARM_STATE_TCP_AFTER_RAW =',after_arm['raw'],flush=True)
    print('JOINT_AFTER_RAW =',after_udp['joint_raw'],flush=True)
    print('UDP_TCP_AFTER_M =',np.round(xyz_m(after_udp['raw']),9).tolist(),flush=True)
    print('ARM_STATE_TCP_AFTER_M =',np.round(xyz_m(after_arm['raw']),9).tolist(),flush=True)
    print('JOINT_AFTER_DEG =',np.round(joint_deg(after_udp['joint_raw']),6).tolist(),flush=True)
    print('UDP_DELTA_MM =',np.round(udp_delta,6).tolist(),flush=True)
    print('ARM_STATE_DELTA_MM =',np.round(arm_delta,6).tolist(),flush=True)
    print('JOINT_DELTA_DEG =',np.round(joint_delta,6).tolist(),flush=True)
    arm_mag=float(np.linalg.norm(arm_delta)); udp_mag=float(np.linalg.norm(udp_delta)); jmag=float(np.linalg.norm(joint_delta))
    if arm_delta[0]>2.0 and udp_delta[0]>2.0 and jmag>0.05 and np.max(np.abs(arm_delta-udp_delta))<1.0: status='PHYSICAL_MOTION_CONFIRMED'
    elif udp_mag>2.0 and arm_mag<0.5 and jmag<0.05: status='UDP_WAYPOINT_NOT_VALID_AS_PHYSICAL_MOTION_FEEDBACK'
    elif arm_mag>2.0 and jmag>0.05 and np.max(np.abs(arm_delta-udp_delta))>1.0: status='UDP_FEEDBACK_MISMATCH'
    elif arm_delta[0]>2.0 and udp_delta[0]<0.5: status='UDP_FEEDBACK_MISMATCH'
    elif udp_delta[0]>2.0 and arm_delta[0]<0.5: status='FEEDBACK_SOURCE_MISMATCH'
    else: status='PHYSICAL_MOTION_NOT_CONFIRMED'
    print('command_count =',command_count,flush=True)
    print('TCP_RESPONSE_TIMEOUT =',str(response_timeout).lower(),flush=True)
    print('final_status =',status,flush=True)
    sock.close(); return 0
if __name__=='__main__': raise SystemExit(main())
