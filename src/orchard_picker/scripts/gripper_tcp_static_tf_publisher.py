#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node

from orchard_picker.protocol import quaternion_from_euler

try:
    from tf2_ros import StaticTransformBroadcaster
except ImportError:  # ROS2 Foxy compatibility
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class GripperTcpStaticTfPublisher(Node):
    def __init__(self):
        super().__init__("gripper_tcp_static_tf_publisher")

        self.enabled = bool(self._param("enabled", True))
        self.parent_frame = self._param("parent_frame", "robot_end")
        self.child_frame = self._param("child_frame", "gripper_tip_center")
        gripper_length = float(self._param("gripper_length_to_tip_m", 0.158))
        adapter_length = float(self._param("flange_adapter_length_m", 0.010))
        if gripper_length < 0.0 or adapter_length < 0.0:
            raise ValueError("gripper and flange adapter lengths must be non-negative")
        self.tcp_xyz = self._vector(
            self._param(
                "gripper_tcp_xyz_m",
                [0.0, 0.0, gripper_length + adapter_length],
            ),
            "gripper_tcp_xyz_m",
        )
        self.tcp_rpy_deg = self._vector(
            self._param("gripper_tcp_rpy_deg", [0.0, 0.0, 0.0]),
            "gripper_tcp_rpy_deg",
        )
        self.republish_period_s = max(
            0.1, float(self._param("republish_period_s", 1.0))
        )

        self.broadcaster = StaticTransformBroadcaster(self)
        self.add_on_set_parameters_callback(self._on_parameters_set)
        self._republish_timer = None
        if not self.enabled:
            self.get_logger().info("Tool center transform publisher disabled")
            return
        self._publish_transform()
        # Periodic resend makes the transient-local static channel visible to
        # late ROS graph participants on Foxy as well as preserving the usual
        # StaticTransformBroadcaster semantics.
        self._republish_timer = self.create_timer(
            self.republish_period_s, self._republish_transform
        )

    def _republish_transform(self):
        self._publish_transform(announce=False)

    def _publish_transform(self, announce=True):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = self.tcp_xyz[0]
        transform.transform.translation.y = self.tcp_xyz[1]
        transform.transform.translation.z = self.tcp_xyz[2]
        quaternion = quaternion_from_euler(
            *[math.radians(value) for value in self.tcp_rpy_deg]
        )
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.broadcaster.sendTransform(transform)
        if announce:
            self.get_logger().info(
                "Published static TF on /tf_static {} -> {}: xyz_m={} rpy_deg={}".format(
                    self.parent_frame,
                    self.child_frame,
                    self._format(self.tcp_xyz),
                    self._format(self.tcp_rpy_deg),
                )
            )

    def _on_parameters_set(self, parameters):
        xyz = list(self.tcp_xyz)
        rpy = list(self.tcp_rpy_deg)
        changed = False
        try:
            for parameter in parameters:
                if parameter.name == "gripper_tcp_xyz_m":
                    xyz = self._vector(parameter.value, parameter.name)
                    if any(abs(value) > 1.0 for value in xyz):
                        raise ValueError("gripper_tcp_xyz_m must be within +/-1 m")
                    changed = True
                elif parameter.name == "gripper_tcp_rpy_deg":
                    rpy = self._vector(parameter.value, parameter.name)
                    if any(abs(value) > 360.0 for value in rpy):
                        raise ValueError("gripper_tcp_rpy_deg must be within +/-360 deg")
                    changed = True
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if changed:
            self.tcp_xyz = xyz
            self.tcp_rpy_deg = rpy
            if self.enabled:
                self._publish_transform()
        return SetParametersResult(successful=True)

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    @staticmethod
    def _vector(values, name):
        result = [float(value) for value in values]
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise ValueError("{} must contain three finite values".format(name))
        return result

    @staticmethod
    def _format(values):
        return "[{}]".format(", ".join("{:.6f}".format(value) for value in values))


def main(args=None):
    rclpy.init(args=args)
    node = GripperTcpStaticTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
