#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node

try:
    from tf2_ros import StaticTransformBroadcaster
except ImportError:  # ROS2 Foxy compatibility
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class HandEyeStaticTfPublisher(Node):
    def __init__(self):
        super().__init__("hand_eye_static_tf_publisher")

        self.enabled = bool(self._param("enabled", True))
        self.parent_frame = self._param("parent_frame", "robot_end")
        self.child_frame = self._param("child_frame", "camera_link")
        self.rotation_matrix = self._rotation_values(
            self._param("rotation_matrix", self._default_rotation())
        )
        self.translation_m = self._translation_values(
            self._param("translation_m", self._default_translation())
        )
        self.invert_matrix = bool(self._param("invert_matrix", False))
        self.republish_period_s = max(
            0.1, float(self._param("republish_period_s", 1.0))
        )

        self.broadcaster = StaticTransformBroadcaster(self)
        self.add_on_set_parameters_callback(self._on_parameters_set)
        self._republish_timer = None
        if not self.enabled:
            self.get_logger().info("Hand-eye static TF publisher disabled")
            return
        self._publish_transform()
        # Foxy's static broadcaster uses transient-local QoS, but a periodic
        # resend keeps the topic discoverable while late graph participants
        # and CLI queries are still joining the DDS graph.
        self._republish_timer = self.create_timer(
            self.republish_period_s, self._republish_transform
        )

    def _republish_transform(self):
        self._publish_transform(announce=False)

    def _publish_transform(self, announce=True):
        rotation = [
            self.rotation_matrix[0:3],
            self.rotation_matrix[3:6],
            self.rotation_matrix[6:9],
        ]
        translation = list(self.translation_m)
        if self.invert_matrix:
            rotation, translation = self._invert_transform(rotation, translation)

        quat = self._quaternion_from_rotation(rotation)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(transform)
        if announce:
            self.get_logger().info(
                "Published static hand-eye TF on /tf_static {} -> {}; translation_m={}".format(
                    self.parent_frame,
                    self.child_frame,
                    self._format(self.translation_m),
                )
            )

    def _on_parameters_set(self, parameters):
        rotation = list(self.rotation_matrix)
        translation = list(self.translation_m)
        changed = False
        try:
            for parameter in parameters:
                if parameter.name == "rotation_matrix":
                    rotation = self._rotation_values(parameter.value)
                    changed = True
                elif parameter.name == "translation_m":
                    translation = self._translation_values(parameter.value)
                    changed = True
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if changed:
            self.rotation_matrix = rotation
            self.translation_m = translation
            if self.enabled:
                self._publish_transform()
        return SetParametersResult(successful=True)

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    @staticmethod
    def _rotation_values(values):
        result = [float(value) for value in values]
        if len(result) != 9 or not all(math.isfinite(value) for value in result):
            raise ValueError("rotation_matrix must contain nine finite values")
        return result

    @staticmethod
    def _translation_values(values):
        result = [float(value) for value in values]
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise ValueError("translation_m must contain three finite values")
        return result

    @staticmethod
    def _format(values):
        return "[{}]".format(
            ", ".join("{:.12g}".format(float(value)) for value in values)
        )

    @staticmethod
    def _default_rotation():
        return [
            -0.85684041,
            -0.51514442,
            0.02123071,
            0.51506258,
            -0.85709906,
            -0.00957856,
            0.02313116,
            0.00272785,
            0.99972872,
        ]

    @staticmethod
    def _default_translation():
        return [0.0757663, 0.0625736, 0.01734084]

    @staticmethod
    def _invert_transform(rotation, translation):
        inverse_rotation = [
            [rotation[0][0], rotation[1][0], rotation[2][0]],
            [rotation[0][1], rotation[1][1], rotation[2][1]],
            [rotation[0][2], rotation[1][2], rotation[2][2]],
        ]
        inverse_translation = [
            -sum(inverse_rotation[row][col] * translation[col] for col in range(3))
            for row in range(3)
        ]
        return inverse_rotation, inverse_translation

    @staticmethod
    def _quaternion_from_rotation(rotation):
        m00, m01, m02 = rotation[0]
        m10, m11, m12 = rotation[1]
        m20, m21, m22 = rotation[2]
        trace = m00 + m11 + m22

        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (m21 - m12) / scale
            qy = (m02 - m20) / scale
            qz = (m10 - m01) / scale
        elif m00 > m11 and m00 > m22:
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / scale
            qx = 0.25 * scale
            qy = (m01 + m10) / scale
            qz = (m02 + m20) / scale
        elif m11 > m22:
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / scale
            qx = (m01 + m10) / scale
            qy = 0.25 * scale
            qz = (m12 + m21) / scale
        else:
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / scale
            qx = (m02 + m20) / scale
            qy = (m12 + m21) / scale
            qz = 0.25 * scale

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        return (qx / norm, qy / norm, qz / norm, qw / norm)


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeStaticTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
