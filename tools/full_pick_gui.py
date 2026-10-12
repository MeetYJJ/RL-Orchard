#!/usr/bin/env python3
"""Small optional PyQt5 front end for tools/launch_full_pick.py."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication, QGridLayout, QLabel, QMainWindow, QPushButton,
        QTextEdit, QWidget,
    )
except ImportError as exc:  # Keep the headless launcher independent of PyQt5.
    raise SystemExit("PyQt5 is required only for the optional GUI: {}".format(exc))

try:
    from launch_full_pick import ProcessManager
except ImportError:
    from tools.launch_full_pick import ProcessManager


class FullPickWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.setWindowTitle("Full Pick V2 Launch Manager")
        self.resize(760, 480)
        self.manager = ProcessManager(config_path)
        self.log_offsets = {}

        central = QWidget(self)
        layout = QGridLayout(central)
        self.bridge_status = QLabel("UNKNOWN")
        self.vision_status = QLabel("UNKNOWN")
        self.full_pick_status = QLabel("STOP")
        layout.addWidget(QLabel("Bridge:"), 0, 0)
        layout.addWidget(self.bridge_status, 0, 1)
        layout.addWidget(QLabel("Vision:"), 1, 0)
        layout.addWidget(self.vision_status, 1, 1)
        layout.addWidget(QLabel("Full Pick:"), 2, 0)
        layout.addWidget(self.full_pick_status, 2, 1)

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.estop_button = QPushButton("急停")
        layout.addWidget(self.start_button, 0, 2)
        layout.addWidget(self.stop_button, 1, 2)
        layout.addWidget(self.estop_button, 2, 2)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 3, 0, 1, 3)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.start_stack)
        self.stop_button.clicked.connect(self.stop_stack)
        self.estop_button.clicked.connect(self.emergency_stop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)

    def append_events(self) -> None:
        while True:
            try:
                line = self.manager.events.get_nowait()
            except Exception:
                return
            self.log_view.append(line)

    def append_process_logs(self) -> None:
        for name in ("bridge", "vision", "full_pick"):
            path = self.manager.log_dir / (name + ".log")
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(self.log_offsets.get(name, 0))
                    for line in stream:
                        self.log_view.append("[{}] {}".format(name.upper(), line.rstrip()))
                    self.log_offsets[name] = stream.tell()
            except OSError:
                continue

    def refresh(self) -> None:
        self.append_events()
        self.append_process_logs()
        self.bridge_status.setText(
            "OK" if self.manager.external.get("bridge") or self.manager.is_alive("bridge") else "FAIL"
        )
        self.vision_status.setText(
            "OK" if self.manager.external.get("vision") or self.manager.is_alive("vision") else "FAIL"
        )
        self.full_pick_status.setText("RUNNING" if self.manager.is_alive("full_pick") else "STOP")

    def start_stack(self) -> None:
        self.start_button.setEnabled(False)
        self.log_view.append("STARTUP_MANAGER_REQUESTED = true")

        def run() -> None:
            ok = self.manager.start_all()
            self.manager.log("STARTUP_MANAGER_RESULT = {}".format(str(ok).lower()))

        threading.Thread(target=run, daemon=True).start()

    def stop_stack(self) -> None:
        threading.Thread(target=self.manager.shutdown, daemon=True).start()

    def emergency_stop(self) -> None:
        def run() -> None:
            self.manager.stop_process("full_pick")
            self.manager._stop_robot()
            self.manager.log("EMERGENCY_STOP_REQUESTED = true")

        threading.Thread(target=run, daemon=True).start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.manager.shutdown()
        event.accept()


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    config = Path(__file__).resolve().parents[1] / "config" / "runtime.yaml"
    window = FullPickWindow(config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
