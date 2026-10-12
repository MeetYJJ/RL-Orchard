from __future__ import annotations
from dataclasses import dataclass
import json, math, socket, time
import numpy as np
from .transforms import realman_udp_euler_to_matrix

LOCAL_UDP = ("192.168.1.100", 8090)
UDP_MAX_AGE = 0.150

@dataclass
class RobotFeedback:
    receipt_mono: float
    joint_rad: np.ndarray
    end_xyz_m: np.ndarray
    euler_raw: tuple[float,float,float]
    err: object = None
    joint_err_code: object = None
    joint_en_flag: object = None
    source: str = ""

    @property
    def fresh(self):
        return time.monotonic() - self.receipt_mono <= UDP_MAX_AGE

    @property
    def healthy(self):
        def zeros(value):
            vals = value if isinstance(value, (list,tuple)) else [value]
            return all(int(v) == 0 for v in vals if v is not None)
        enabled = self.joint_en_flag is None or all(bool(int(v)) for v in self.joint_en_flag)
        try:
            realman_udp_euler_to_matrix(self.euler_raw)
            rotation_valid = True
        except (TypeError, ValueError):
            rotation_valid = False
        return self.fresh and np.all(np.isfinite(self.joint_rad)) and np.all(np.isfinite(self.end_xyz_m)) and rotation_valid and zeros(self.err) and zeros(self.joint_err_code) and enabled

class UdpFeedback:
    def __init__(self, bind=LOCAL_UDP):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(bind)
        self.sock.setblocking(False)
        self.latest: RobotFeedback | None = None
        self.history: list[RobotFeedback] = []
        self.malformed = 0
        self.last_malformed_error: str | None = None

    @staticmethod
    def decode(data, src=""):
        p = json.loads(data.decode("utf-8").strip())
        js, wp = p.get("joint_status"), p.get("waypoint")
        jp, pos, euler = (js or {}).get("joint_position"), (wp or {}).get("position"), (wp or {}).get("euler")
        if not (isinstance(jp,list) and len(jp)==6 and isinstance(pos,list) and len(pos)==3 and isinstance(euler,list) and len(euler)==3):
            raise ValueError("invalid UDP feedback dimensions")
        joint_rad = np.asarray([float(v)*0.001*math.pi/180 for v in jp], dtype=np.float64)
        end_xyz_m = np.asarray(pos, dtype=np.float64)/1e6
        euler_raw = np.asarray(euler, dtype=np.float64)
        if not np.all(np.isfinite(joint_rad)) or not np.all(np.isfinite(end_xyz_m)):
            raise ValueError("invalid UDP numeric feedback")
        # Validate the exact orientation conversion before accepting this as
        # latest.  A bad frame cannot replace the last known-good pose.
        realman_udp_euler_to_matrix(euler_raw)
        return RobotFeedback(time.monotonic(), joint_rad, end_xyz_m, tuple(float(v) for v in euler_raw), p.get("err"), js.get("joint_err_code"), js.get("joint_en_flag"), str(src))

    def poll(self):
        while True:
            try:
                data, src = self.sock.recvfrom(65535)
            except BlockingIOError:
                return self.latest
            except OSError:
                return self.latest
            try:
                value = self.decode(data, src)
                self.latest = value
                self.history.append(value)
                if len(self.history) > 200:
                    self.history.pop(0)
            except Exception as exc:
                self.malformed += 1
                self.last_malformed_error = f"{type(exc).__name__}:{exc}"
        return self.latest

    def close(self):
        self.sock.close()
