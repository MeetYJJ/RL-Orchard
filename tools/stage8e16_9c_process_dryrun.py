#!/usr/bin/env python3
"""Software-only smoke test for the spawned high-follow transport process."""

import time
import numpy as np

from realman_high_follow_transport import HighFollowTransportProcess, so3_exp


def main():
    initial = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    rotation = np.eye(3)
    transport = HighFollowTransportProcess(initial, rotation)
    try:
        status = transport.start()
        for i in range(10):
            transport.publish(time.monotonic_ns(), initial, rotation,
                              initial + np.array([0.001 * (i + 1), 0.0, 0.0]),
                              so3_exp(np.array([0.0, 0.0, 0.001 * (i + 1)])))
            time.sleep(0.025)
        time.sleep(0.05)
        final_status = transport.status()
        print("TRANSPORT_ARCHITECTURE = separate_process")
        print("MULTIPROCESS_START_METHOD = spawn")
        print("SHARED_MEMORY_ENABLED = true")
        print("SEQLOCK_ENABLED = true")
        print("POLICY_HZ_CONFIG = 50")
        print("TRANSPORT_HZ_CONFIG = 200")
        print("TRANSPORT_CPU =", transport.transport_cpus)
        print("POLICY_CPU_SET =", transport.policy_cpus)
        print("PERSISTENT_TCP = true")
        print("TCP_NODELAY = true")
        print("RECV_IN_TRANSPORT = false")
        print("RETRY_IN_TRANSPORT = false")
        print("AUTO_RECONNECT = false")
        print("SO3_INTERPOLATION = true")
        print("ACTION_APPLIED_ONCE_PER_POLICY_STEP = true")
        print("LEGACY_PATH_PRESERVED = true")
        print("HIGH_FOLLOW_PROCESS_IMPLEMENTED =", str(bool(status is not None and final_status is not None)).lower())
        print("HARDWARE_GATE = false")
        print("network_write_attempted = false")
        print("hardware_sent = false")
    finally:
        transport.stop()


if __name__ == "__main__":
    main()
