#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from apple_interfaces.msg import ApplePositionArray


class AppleListener(Node):
    """Small receiver used to verify the published custom message."""

    def __init__(self):
        super().__init__('apple_listener')
        self.subscription = self.create_subscription(
            ApplePositionArray,
            '/apple_positions',
            self.on_message,
            10)

    def on_message(self, message):
        stamp = message.header.stamp
        if not message.apples:
            self.get_logger().info(
                'time={0}.{1:09d}, no apple'.format(
                    stamp.sec, stamp.nanosec))
            return

        for apple in message.apples:
            self.get_logger().info(
                'time={0}.{1:09d}, id={2}, '
                'x={3:.4f}, y={4:.4f}, z={5:.4f} m, frame={6}'.format(
                    stamp.sec,
                    stamp.nanosec,
                    apple.id,
                    apple.x,
                    apple.y,
                    apple.z,
                    message.header.frame_id))


def main(args=None):
    rclpy.init(args=args)
    node = AppleListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
