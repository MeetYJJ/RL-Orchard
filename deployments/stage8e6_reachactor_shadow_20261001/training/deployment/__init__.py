"""Hardware-independent deployment primitives.

Modules in this package intentionally depend only on NumPy.  They do not
open sockets, import ROS/MuJoCo/PyTorch, or construct RealMan protocol
messages.
"""

from .cartesian_controller_core import (
    CartesianControllerConfig,
    CartesianControllerCore,
    ControllerResult,
    Pose,
    RobotState,
)

__all__ = [
    "CartesianControllerConfig",
    "CartesianControllerCore",
    "ControllerResult",
    "Pose",
    "RobotState",
]
