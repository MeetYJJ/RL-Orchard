from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('apple_vision'),
        'config',
        'apple_detector.yaml')

    return LaunchDescription([
        Node(
            package='apple_vision',
            executable='apple_detector',
            name='apple_detector',
            output='screen',
            parameters=[config],
        )
    ])
