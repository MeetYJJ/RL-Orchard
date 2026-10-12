#!/usr/bin/env python3

import json
import os
import threading
import time
from array import array

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
from geometry_msgs.msg import PointStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

try:
    from rclpy.qos import DurabilityPolicy
except ImportError:  # ROS 2 Foxy compatibility
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
from ultralytics import SAM, YOLO

from apple_interfaces.msg import ApplePosition, ApplePositionArray


class AppleDetectorNode(Node):
    """Detect tracked apples and publish their 3D camera coordinates."""

    def __init__(self):
        super().__init__('apple_detector')

        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('sam_model_path', '')
        self.declare_parameter('confidence_threshold', 0.9)
        self.declare_parameter('enable_sam', False)
        self.declare_parameter('sam_every_n_frames', 3)
        # The PyQt GUI subscribes to the annotated image topic.  Keep OpenCV's
        # separate desktop window disabled unless explicitly requested.
        self.declare_parameter('show_image', False)
        self.declare_parameter('publish_topic', '/apple_positions')
        self.declare_parameter('publish_target_point', True)
        self.declare_parameter('target_point_topic', '/apple/target_point')
        self.declare_parameter(
            'annotated_image_topic', '/apple/annotated_image')
        self.declare_parameter('target_info_topic', '/apple/target_info')
        self.declare_parameter('raw_rgb_topic', '/apple_vision/rgb')
        self.declare_parameter(
            'aligned_depth_topic', '/apple_vision/aligned_depth')
        self.declare_parameter(
            'camera_metadata_topic', '/apple_vision/camera_metadata')
        self.declare_parameter('training_publish_hz', 30.0)
        self.declare_parameter('target_apple_id', -1)
        self.declare_parameter(
            'camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('camera_serial', '')
        self.declare_parameter('timer_period', 0.01)
        self.declare_parameter('depth_window_radius', 3)
        self.declare_parameter('apple_radius_m', 0.03)
        self.declare_parameter('target_stability_frames', 5)
        self.declare_parameter('target_max_jitter_m', 0.015)
        self.declare_parameter('target_min_distance_m', 0.15)
        self.declare_parameter('target_max_distance_m', 1.20)

        yolo_path = self.get_parameter(
            'yolo_model_path').get_parameter_value().string_value
        sam_path = self.get_parameter(
            'sam_model_path').get_parameter_value().string_value
        self.confidence = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value
        self.add_on_set_parameters_callback(self._on_parameters_set)
        self.enable_sam = self.get_parameter(
            'enable_sam').get_parameter_value().bool_value
        self.sam_every_n_frames = max(
            1,
            self.get_parameter(
                'sam_every_n_frames').get_parameter_value().integer_value)
        self.show_image = self.get_parameter(
            'show_image').get_parameter_value().bool_value
        publish_topic = self.get_parameter(
            'publish_topic').get_parameter_value().string_value
        self.publish_target_point = self.get_parameter(
            'publish_target_point').get_parameter_value().bool_value
        target_point_topic = self.get_parameter(
            'target_point_topic').get_parameter_value().string_value
        annotated_image_topic = self.get_parameter(
            'annotated_image_topic').get_parameter_value().string_value
        target_info_topic = self.get_parameter(
            'target_info_topic').get_parameter_value().string_value
        raw_rgb_topic = self.get_parameter(
            'raw_rgb_topic').get_parameter_value().string_value
        aligned_depth_topic = self.get_parameter(
            'aligned_depth_topic').get_parameter_value().string_value
        camera_metadata_topic = self.get_parameter(
            'camera_metadata_topic').get_parameter_value().string_value
        training_publish_hz = self.get_parameter(
            'training_publish_hz').get_parameter_value().double_value
        self.training_publish_period = (
            1.0 / training_publish_hz if training_publish_hz > 0.0 else 0.0)
        self.target_apple_id = self.get_parameter(
            'target_apple_id').get_parameter_value().integer_value
        self.camera_frame = self.get_parameter(
            'camera_frame').get_parameter_value().string_value
        camera_serial = self.get_parameter(
            'camera_serial').get_parameter_value().string_value.strip()
        timer_period = self.get_parameter(
            'timer_period').get_parameter_value().double_value
        self.depth_window_radius = max(
            0,
            self.get_parameter(
                'depth_window_radius').get_parameter_value().integer_value)
        self.apple_radius_m = max(
            0.0,
            self.get_parameter(
                'apple_radius_m').get_parameter_value().double_value)
        self.target_stability_frames = max(
            1,
            self.get_parameter(
                'target_stability_frames').get_parameter_value().integer_value)
        self.target_max_jitter_m = max(
            0.0,
            self.get_parameter(
                'target_max_jitter_m').get_parameter_value().double_value)
        self.target_min_distance_m = max(
            0.0,
            self.get_parameter(
                'target_min_distance_m').get_parameter_value().double_value)
        self.target_max_distance_m = max(
            self.target_min_distance_m,
            self.get_parameter(
                'target_max_distance_m').get_parameter_value().double_value)

        if not yolo_path:
            raise RuntimeError(
                'Parameter "yolo_model_path" must point to the YOLO weights.')
        if not os.path.isfile(yolo_path):
            raise FileNotFoundError(
                'YOLO model was not found: {0}'.format(yolo_path))
        if self.enable_sam and not sam_path:
            raise RuntimeError(
                'Parameter "sam_model_path" is required when enable_sam=true.')
        if self.enable_sam and not os.path.isfile(sam_path):
            raise FileNotFoundError(
                'SAM model was not found: {0}'.format(sam_path))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(
            'Loading YOLO model on {0}: {1}'.format(
                self.device, yolo_path))
        self.yolo_model = YOLO(yolo_path)
        self.sam_model = SAM(sam_path) if self.enable_sam else None

        self.publisher = self.create_publisher(
            ApplePositionArray, publish_topic, 10)
        self.target_point_publisher = self.create_publisher(
            PointStamped, target_point_topic, 10)
        self.annotated_image_publisher = self.create_publisher(
            CompressedImage, annotated_image_topic, 2)
        metadata_qos = QoSProfile(depth=1)
        metadata_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.target_info_publisher = self.create_publisher(
            String, target_info_topic, metadata_qos)
        self.raw_rgb_publisher = self.create_publisher(
            Image, raw_rgb_topic, qos_profile_sensor_data)
        self.aligned_depth_publisher = self.create_publisher(
            Image, aligned_depth_topic, qos_profile_sensor_data)
        self.camera_metadata_publisher = self.create_publisher(
            String, camera_metadata_topic, metadata_qos)

        self.pipeline = rs.pipeline()
        config = rs.config()
        if camera_serial:
            config.enable_device(camera_serial)
        config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(
            rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline_started = False
        pipeline_profile = self.pipeline.start(config)
        self.pipeline_started = True
        camera_device = pipeline_profile.get_device()
        camera_name = camera_device.get_info(rs.camera_info.name)
        selected_serial = camera_device.get_info(rs.camera_info.serial_number)
        firmware = camera_device.get_info(rs.camera_info.firmware_version)
        self.camera_name = str(camera_name)
        self.camera_serial = str(selected_serial)
        self.depth_scale_m = float(
            camera_device.first_depth_sensor().get_depth_scale())
        self.get_logger().info(
            'RealSense started: {0}, serial={1}, firmware={2}'.format(
                camera_name, selected_serial, firmware))

        self.frame_count = 0
        self.last_masks_by_id = {}
        self.target_history_id = None
        self.target_history = []
        self.confidence_by_id = {}
        self.last_process_started = None

        # One acquisition thread owns the single RealSense pipeline.  YOLO
        # and SAM consume only the newest cached frameset, so inference cannot
        # delay raw RGB/depth publication.
        self._camera_condition = threading.Condition()
        self._camera_stop = threading.Event()
        self._latest_camera_frame = None
        self._camera_sequence = 0
        self._last_inference_sequence = -1
        self._last_published_sequence = -1
        self._last_training_publish = 0.0
        self._camera_ready_logged = False
        self._raw_publish_ready_logged = False
        self._camera_rate_started = None
        self._camera_rate_count = 0
        self._camera_rate_logged = False
        self._raw_rate_started = None
        self._raw_rate_count = 0
        self._raw_rate_logged = False
        self._camera_thread = threading.Thread(
            target=self._camera_acquisition_loop,
            name='realsense-acquisition',
            daemon=True)
        self._raw_publisher_thread = threading.Thread(
            target=self._raw_publisher_loop,
            name='raw-frame-publisher',
            daemon=True)
        self._camera_thread.start()
        self._raw_publisher_thread.start()
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.camera_metadata_publisher.publish(
            String(data=json.dumps({
                'camera_model': self.camera_name,
                'camera_serial': self.camera_serial,
                'depth_scale_m': self.depth_scale_m,
                'depth_unit': 'device_unit',
                'rgb_encoding': 'bgr8',
                'depth_encoding': '16UC1',
                'aligned_depth_to_color': True,
                'width': 640,
                'height': 480,
            }, separators=(',', ':'))))

        if self.show_image:
            cv2.namedWindow('Apple detector', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            'Apple detector started; publishing on {0}'.format(
                publish_topic))
        if self.publish_target_point:
            self.get_logger().info(
                'Publishing picker target point on {0}'.format(
                    target_point_topic))

    def _on_parameters_set(self, parameters):
        for parameter in parameters:
            if parameter.name != 'confidence_threshold':
                continue
            try:
                confidence = float(parameter.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason='confidence_threshold must be a number')
            if not 0.0 <= confidence <= 1.0:
                return SetParametersResult(
                    successful=False,
                    reason='confidence_threshold must be in [0, 1]')
            self.confidence = confidence
        return SetParametersResult(successful=True)

    def process_frame(self):
        self.last_process_started = time.monotonic()

        with self._camera_condition:
            frame = self._latest_camera_frame

        if frame is None:
            return

        sequence = frame['sequence']
        if sequence == self._last_inference_sequence:
            return
        self._last_inference_sequence = sequence

        color_image = frame['color_image']
        depth_image = frame['depth_image']
        color_intrinsics = frame['color_intrinsics']
        depth_scale_m = frame['depth_scale_m']

        results = self.yolo_model.track(
            color_image,
            conf=self.confidence,
            device=self.device,
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False)

        detections = self._extract_detections(results)
        masks = self._get_masks(color_image, detections)
        message = ApplePositionArray()

        # Preserve the existing target/detection publication semantics: this
        # timestamp is when its result is assembled. Raw RGB/depth carry the
        # source frameset timestamp from the acquisition thread instead.
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.camera_frame

        display_image = color_image.copy()
        current_confidence_by_id = {}

        for detection in detections:
            track_id = detection['id']
            if track_id < 0:
                continue

            x1, y1, x2, y2 = detection['box']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            depth = self._get_depth(
                depth_image,
                center_x,
                center_y,
                masks.get(track_id),
                depth_scale_m)
            if depth <= 0.0:
                continue

            point = rs.rs2_deproject_pixel_to_point(
                color_intrinsics, [center_x, center_y], depth)

            apple = ApplePosition()
            apple.id = int(track_id)
            apple.x = float(point[0])
            apple.y = float(point[1])
            apple.z = float(point[2])
            message.apples.append(apple)
            current_confidence_by_id[track_id] = float(
                detection['confidence'])

            self._draw_detection(
                display_image, detection, apple, center_x, center_y)

        self.confidence_by_id = current_confidence_by_id

        # An empty array is also published. The receiver can then distinguish
        # "no apples in this frame" from "the detector stopped publishing".
        self.publisher.publish(message)
        self._publish_target_point(message, detections)
        self._publish_annotated_image(display_image, message.header)

        if self.show_image:
            cv2.imshow('Apple detector', display_image)
            cv2.waitKey(1)

        self.frame_count += 1

    def _camera_acquisition_loop(self):
        """Read and align frames without running inference in that thread."""
        while not self._camera_stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Copy while the RealSense frame is owned by this thread. A
                # single cache entry bounds memory and drops stale frames when
                # detection is slower than the camera.
                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                color_intrinsics = (
                    color_frame.profile.as_video_stream_profile().intrinsics)
                stamp = self.get_clock().now().to_msg()
                with self._camera_condition:
                    self._camera_sequence += 1
                    self._latest_camera_frame = {
                        'sequence': self._camera_sequence,
                        'color_image': color_image,
                        'depth_image': depth_image,
                        'color_intrinsics': color_intrinsics,
                        'depth_scale_m': self.depth_scale_m,
                        'stamp': stamp,
                    }
                    if not self._camera_ready_logged:
                        self._camera_ready_logged = True
                        self.get_logger().info(
                            'RealSense acquisition cache is receiving frames')
                    if self._camera_rate_started is None:
                        self._camera_rate_started = time.monotonic()
                    self._camera_rate_count += 1
                    if self._camera_rate_count == 30 \
                            and not self._camera_rate_logged:
                        elapsed = time.monotonic() - self._camera_rate_started
                        if elapsed > 0.0:
                            self._camera_rate_logged = True
                            self.get_logger().info(
                                'RealSense cache rate: {0:.2f} Hz'.format(
                                    self._camera_rate_count / elapsed))
                    self._camera_condition.notify_all()
            except Exception as exc:
                if self._camera_stop.is_set():
                    break
                self.get_logger().warning(
                    'RealSense acquisition failed: {0}'.format(exc))
                self._camera_stop.wait(0.05)

    def _raw_publisher_loop(self):
        """Publish newest cached RGB-D pairs independently of YOLO latency."""
        while not self._camera_stop.is_set():
            with self._camera_condition:
                self._camera_condition.wait_for(
                    lambda: self._camera_stop.is_set()
                    or (
                        self._latest_camera_frame is not None
                        and self._latest_camera_frame['sequence']
                        > self._last_published_sequence),
                    timeout=0.5)
                if self._camera_stop.is_set():
                    break
                frame = self._latest_camera_frame

            if frame is None:
                continue

            # Rate-limit without building a queue of stale image frames.
            period = self.training_publish_period
            if period > 0.0:
                remaining = period - (
                    time.monotonic() - self._last_training_publish)
                if remaining > 0.0:
                    self._camera_stop.wait(remaining)
                    if self._camera_stop.is_set():
                        break
                    with self._camera_condition:
                        frame = self._latest_camera_frame
                    if frame is None:
                        continue

            sequence = frame['sequence']
            if sequence <= self._last_published_sequence:
                continue
            try:
                self._publish_training_frames(
                    frame['color_image'],
                    frame['depth_image'],
                    frame['stamp'])
                if not self._raw_publish_ready_logged:
                    self._raw_publish_ready_logged = True
                    self.get_logger().info(
                        'Independent raw RGB/depth publisher is active')
                if self._raw_rate_started is None:
                    self._raw_rate_started = time.monotonic()
                self._raw_rate_count += 1
                if self._raw_rate_count == 30 and not self._raw_rate_logged:
                    elapsed = time.monotonic() - self._raw_rate_started
                    if elapsed > 0.0:
                        self._raw_rate_logged = True
                        self.get_logger().info(
                            'Raw RGB/depth publish call rate: {0:.2f} Hz'
                            .format(self._raw_rate_count / elapsed))
                self._last_training_publish = time.monotonic()
                self._last_published_sequence = sequence
            except Exception as exc:
                self.get_logger().warning(
                    'Training-frame publish failed: {0}'.format(exc))

    @staticmethod
    def _extract_detections(results):
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [
                    int(value) for value in box.xyxy[0].tolist()]
                track_id = (
                    int(box.id[0].item()) if box.id is not None else -1)
                detections.append({
                    'id': track_id,
                    'confidence': float(box.conf[0].item()),
                    'box': [x1, y1, x2, y2],
                })
        return detections

    def _get_masks(self, color_image, detections):
        if not self.enable_sam or not detections:
            return {}

        if self.frame_count % self.sam_every_n_frames != 0:
            return {
                detection['id']: self.last_masks_by_id[detection['id']]
                for detection in detections
                if detection['id'] in self.last_masks_by_id
            }

        boxes = [detection['box'] for detection in detections]
        results = self.sam_model(
            color_image,
            bboxes=boxes,
            device=self.device,
            verbose=False)
        if not results or results[0].masks is None:
            return {}

        masks = {}
        mask_data = results[0].masks.data.cpu().numpy()
        for detection, raw_mask in zip(detections, mask_data):
            track_id = detection['id']
            if track_id < 0:
                continue
            mask = cv2.resize(
                raw_mask.astype(np.uint8),
                (color_image.shape[1], color_image.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[track_id] = mask
            self.last_masks_by_id[track_id] = mask
        return masks

    def _get_depth(
            self, depth_image, center_x, center_y, mask, depth_scale_m):
        """Return median aligned depth in metres from the cached frame."""
        if mask is not None:
            values = depth_image[mask]
            values = values[values > 0]
            if values.size:
                return float(np.median(values) * depth_scale_m)

        radius = self.depth_window_radius
        height, width = depth_image.shape
        x1 = max(0, center_x - radius)
        x2 = min(width, center_x + radius + 1)
        y1 = max(0, center_y - radius)
        y2 = min(height, center_y + radius + 1)
        values = depth_image[y1:y2, x1:x2]
        values = values[values > 0]
        if values.size:
            return float(np.median(values) * depth_scale_m)
        if 0 <= center_x < width and 0 <= center_y < height:
            return float(depth_image[center_y, center_x] * depth_scale_m)
        return 0.0

    @staticmethod
    def _draw_detection(image, detection, apple, center_x, center_y):
        x1, y1, x2, y2 = detection['box']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = (
            'ID {0} {1:.2f} X:{2:.3f} Y:{3:.3f} Z:{4:.3f}m'.format(
                apple.id,
                detection['confidence'],
                apple.x,
                apple.y,
                apple.z))
        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2)
        cv2.circle(image, (center_x, center_y), 4, (255, 0, 0), -1)

    def _publish_annotated_image(self, image, header):
        success, encoded = cv2.imencode(
            '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            self.get_logger().warning('Failed to encode annotated camera frame')
            return
        message = CompressedImage()
        message.header = header
        message.format = 'jpeg'
        message.data = encoded.tobytes()
        self.annotated_image_publisher.publish(message)

    def _publish_training_frames(self, color_image, depth_image, stamp):
        """Publish the already-opened, depth-aligned RealSense frame pair."""
        # Always call publish.  ROS drops best-effort samples when there are
        # no subscribers; checking subscription_count here is unsafe for a
        # publisher thread because graph updates are executor-driven.  This
        # keeps the raw stream independent of the YOLO callback and makes a
        # subscriber receive the next complete frameset immediately.
        color = np.ascontiguousarray(color_image)
        rgb_message = Image()
        rgb_message.header.stamp = stamp
        rgb_message.header.frame_id = self.camera_frame
        rgb_message.height = int(color.shape[0])
        rgb_message.width = int(color.shape[1])
        rgb_message.encoding = 'bgr8'
        rgb_message.is_bigendian = False
        rgb_message.step = int(color.shape[1] * 3)
        # ROS 2 Foxy generates Image.data as array('B'). Assigning raw bytes
        # causes a slow Python per-byte conversion on this device; wrapping
        # the same bytes in array('B') keeps the copy in C and preserves the
        # exact bgr8 payload.
        rgb_message.data = array('B', color.tobytes())
        self.raw_rgb_publisher.publish(rgb_message)

        depth = np.ascontiguousarray(depth_image, dtype=np.uint16)
        depth_message = Image()
        depth_message.header.stamp = stamp
        depth_message.header.frame_id = self.camera_frame
        depth_message.height = int(depth.shape[0])
        depth_message.width = int(depth.shape[1])
        depth_message.encoding = '16UC1'
        depth_message.is_bigendian = False
        depth_message.step = int(depth.shape[1] * 2)
        depth_message.data = array('B', depth.tobytes())
        self.aligned_depth_publisher.publish(depth_message)



    def _publish_target_point(self, message, detections):

        if not self.publish_target_point or not message.apples:
            self._reset_target_history()
            return



        target = self._select_target_apple(message.apples)

        if target is None:
            self._reset_target_history()
            return

        surface = np.array([target.x, target.y, target.z], dtype=np.float64)
        distance = float(np.linalg.norm(surface))
        if (
                not np.isfinite(distance)
                or distance < self.target_min_distance_m
                or distance > self.target_max_distance_m):
            self._reset_target_history()
            return

        # RealSense measures the visible apple surface.  Approximate the fruit
        # as a sphere and move one radius farther along the camera ray so the
        # published target represents the sphere center.
        center = surface * ((distance + self.apple_radius_m) / distance)

        if self.target_history_id != target.id:
            self.target_history_id = target.id
            self.target_history = []
        self.target_history.append(center)
        self.target_history = self.target_history[-self.target_stability_frames:]
        if len(self.target_history) < self.target_stability_frames:
            return

        samples = np.stack(self.target_history)
        stable_center = samples.mean(axis=0)
        jitter = np.linalg.norm(samples - stable_center, axis=1)
        if self.target_max_jitter_m > 0.0 and float(jitter.max()) > self.target_max_jitter_m:
            return

        point = PointStamped()
        point.header = message.header
        point.point.x = float(stable_center[0])
        point.point.y = float(stable_center[1])
        point.point.z = float(stable_center[2])
        self.target_point_publisher.publish(point)

        target_info = {
            'id': int(target.id),
            'confidence': float(
                self.confidence_by_id.get(target.id, 0.0)),
            'camera_xyz_m': [float(value) for value in stable_center],
            'camera_frame': message.header.frame_id,
            'bbox_xyxy': next(
                (list(item['box']) for item in detections
                 if item['id'] == target.id),
                [-1, -1, -1, -1]),
            'target_valid': True,
        }
        self.target_info_publisher.publish(
            String(data=json.dumps(target_info, separators=(',', ':'))))

    def _reset_target_history(self):
        self.target_history_id = None
        self.target_history = []



    def _select_target_apple(self, apples):

        if self.target_apple_id >= 0:

            for apple in apples:

                if apple.id == self.target_apple_id:

                    return apple

            return None



        return min(apples, key=lambda apple: apple.z)



    def destroy_node(self):
        if hasattr(self, '_camera_stop'):
            self._camera_stop.set()
            with self._camera_condition:
                self._camera_condition.notify_all()
        if getattr(self, 'pipeline_started', False):
            self.pipeline.stop()
            self.pipeline_started = False
        for thread_name in ('_camera_thread', '_raw_publisher_thread'):
            thread = getattr(self, thread_name, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        if getattr(self, 'show_image', False):
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AppleDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(str(exc))
        else:
            print('Failed to start apple_detector: {0}'.format(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
