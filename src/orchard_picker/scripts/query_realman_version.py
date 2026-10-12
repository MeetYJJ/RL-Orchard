#!/usr/bin/env python3
"""Read-only, allowlisted RealMan version query helper.

This deliberately refuses arbitrary JSON and never creates a motion request.
It is an operator tool; it is not used by the recorder.
"""

import argparse
import json

try:
    import rclpy
    from orchard_picker.srv import RawJson
    from rclpy.node import Node
except ImportError:  # permit pure response-validation unit tests before ROS build
    rclpy = None
    RawJson = None
    Node = object


READ_ONLY_COMMANDS = {
    "get_arm_software_version",
    "get_arm_software_info",
    "get_joint_software_version",
}


def validate_version_response(command, response_json):
    """Return the matched version payload or raise on an unrelated response."""
    try:
        bundle = json.loads(response_json)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("controller returned invalid JSON: {}".format(exc))
    ack = bundle.get("ack") if isinstance(bundle, dict) else None
    if not isinstance(ack, dict):
        raise RuntimeError("controller response has no version ack")
    if command == "get_arm_software_version":
        matched = (
            ack.get("state") == "arm_software_version"
            or "Ctrl_version" in ack
            or "Plan_version" in ack
        )
    elif command == "get_arm_software_info":
        matched = (
            ack.get("command") == "arm_software_info"
            or ("Product_version" in ack and "ctrl_info" in ack)
        )
    else:
        matched = (
            ack.get("state") == "joint_software_version"
            and "version" in ack
        )
    if not matched:
        raise RuntimeError(
            "received an unrelated/unsupported response for {}".format(command)
        )
    return ack


class VersionQuery(Node):
    def __init__(self, service_name, command):
        if RawJson is None:
            raise RuntimeError("ROS generated interfaces are unavailable")
        super().__init__("realman_version_query")
        self.client = self.create_client(RawJson, service_name)
        self.command = command
        self.response = None

    def run(self, timeout_sec):
        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("service is unavailable: {}".format(self.client.srv_name))
        request = RawJson.Request()
        request.request_json = json.dumps({"command": self.command})
        request.wait_for_response = True
        request.wait_for_trajectory = False
        request.wait_device = 0
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=float(timeout_sec)
        )
        if not future.done():
            raise RuntimeError("timed out waiting for version response")
        self.response = future.result()
        return self.response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(READ_ONLY_COMMANDS))
    parser.add_argument(
        "--service", default="/rm_json_bridge/raw_json",
        help="existing bridge RawJson service (read-only query only)",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)
    if rclpy is None or RawJson is None:
        print("ROS generated interfaces are unavailable")
        return 2
    rclpy.init()
    node = VersionQuery(args.service, args.command)
    try:
        response = node.run(args.timeout)
        version_payload = validate_version_response(
            args.command, response.response_json
        )
        print(json.dumps({
            "command": args.command,
            "success": bool(response.success),
            "message": str(response.message),
            "version": version_payload,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if response.success else 1
    except Exception as exc:
        print("read-only version query failed: {}".format(exc))
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
