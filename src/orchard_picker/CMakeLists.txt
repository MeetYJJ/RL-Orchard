cmake_minimum_required(VERSION 3.5)
project(orchard_picker)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ExpertAction.msg"
  "msg/RobotState.msg"
  "msg/TaskOutcome.msg"
  "srv/GetArmState.srv"
  "srv/GripperPick.srv"
  "srv/GripperPosition.srv"
  "srv/GripperRelease.srv"
  "srv/MoveJ.srv"
  "srv/MovePose.srv"
  "srv/RawJson.srv"
  "srv/SetToolVoltage.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/fake_apple_target.py
  scripts/gripper_tcp_static_tf_publisher.py
  scripts/hand_eye_static_tf_publisher.py
  scripts/pick_task_manager.py
  scripts/raw_json_cli.py
  scripts/rl_data_recorder.py
  scripts/rm_json_bridge.py
  scripts/inspect_rl_dataset.py
  scripts/label_episode.py
  scripts/replay_episode.py
  scripts/report_collection_coverage.py
  scripts/rl_collection_orchestrator.py
  scripts/check_real_data_pipeline.py
  scripts/query_realman_version.py
  scripts/t930_arm_net_check.sh
  scripts/tcp_tf_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
