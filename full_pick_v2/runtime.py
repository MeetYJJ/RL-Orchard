from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path
from .config import VISION_SCRIPT

class ManagedProcess:
    def __init__(self, name, process, log):
        self.name, self.process, self.log = name, process, log

class ProcessManager:
    def __init__(self, log_dir="/tmp/rl_full_pick_v2"):
        self.log_dir=Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.started=[]
    def start_vision(self):
        log=(self.log_dir/"vision.log").open("ab")
        env=os.environ.copy(); env["CAMERA_SERIAL"]=env.get("CAMERA_SERIAL","908212070350"); env["SHOW_IMAGE"]="false"
        p=subprocess.Popen(["setsid","bash",str(VISION_SCRIPT)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=False)
        self.started.append(ManagedProcess("vision",p,log)); return p
    def start_bridge(self, config):
        log=(self.log_dir/"bridge.log").open("ab")
        p=subprocess.Popen(["setsid","ros2","launch","orchard_picker","driver_only.launch.py",f"config:={config}"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
        self.started.append(ManagedProcess("bridge",p,log)); return p
    def cleanup(self):
        for item in reversed(self.started):
            p=item.process
            if p.poll() is not None: item.log.close(); continue
            try: os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError: pass
            deadline=time.monotonic()+3
            while p.poll() is None and time.monotonic()<deadline: time.sleep(.1)
            if p.poll() is None:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            item.log.close()
        self.started.clear()
