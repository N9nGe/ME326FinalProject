#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from typing import Tuple
from scipy.spatial.transform import Rotation as R

import cv2
import os
import numpy as np
import io
from PIL import Image as PILImage
from ultralytics import YOLO

import geometry_msgs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
from rclpy.duration import Duration

class Camera(Node):
    def __init__(self):
        super().__init__('rohan')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/locobot/camera/camera/color/image_raw',
            self.camera_listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription_depth = self.create_subscription(
            Image,
            '/locobot/camera/camera/depth/image_rect_raw',
            self.depth_listener_callback,
            10)
        self.subscription_depth  # prevent unused variable warning

        self.subscription_audio = self.create_subscription(
            String,
            'AudioItem',  # CHANGE THIS TO THE CORRECT TOPIC
            self.audio_listener_callback,
            10)
        self.subscription_audio  # prevent unused variable warning

        self.publish_target = self.create_publisher(
            PoseStamped,
            '/arm_pose',  # CHANGE THIS TO THE CORRECT TOPIC
            10)
        self.publish_target  # prevent unused variable warning

        # Intrinsics for RGB and Depth cameras
        self.rgb_K = (609.03759765625, 609.2069091796875, 323.6105041503906, 243.78759765625)
        self.depth_K = (379.8849792480469, 379.8849792480469, 320.3520202636719, 379.8849792480469)

        # Identity transform between cameras (fill in if needed)
        self.cam2cam_transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.center_coordinates = None  # Will store best banana center
        self.object = None
        self.rgb = None
        self.rgb_width = 640
        self.rgb_height = 480

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def audio_listener_callback(self, msg):
        self.object = msg.data

    def camera_listener_callback(self, msg):
        self.get_logger().info('Received an image')

        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return

        try:
            # Convert ROS image to OpenCV format (BGR by default)
            cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_ColorImage_rgb = cv2.cvtColor(cv_ColorImage, cv2.COLOR_BGR2RGB)

            # 1) Run YOLO on the frame, get annotated images & banana center
            all_detections_frame, banana_frame, banana_center = self.run_yolo_and_get_images(
                cv_ColorImage_rgb,
                model_path="yolo11x.pt"  # or your custom YOLO weights
            )

            # 2) Save the "all detections" image always
            cv2.imwrite("camera_detections.jpg", all_detections_frame)
            self.get_logger().info("Saved all detections to camera_detections.jpg")

            # 3) If a banana was found, save the banana-only image
            self.center_coordinates = banana_center
            if banana_center != (None, None) and banana_frame is not None:
                cv2.imwrite("banana_detection.jpg", banana_frame)
                self.get_logger().info("Banana found! Saved banana-only image to banana_detection.jpg")
            else:
                self.get_logger().info("No banana detected in the current frame.")

            # Keep the latest RGB frame for depth alignment
            self.rgb = cv_ColorImage

        except Exception as e:
            self.get_logger().error(f"Failed to process RGB image: {str(e)}")

    def depth_listener_callback(self, msg):
        """
        When a new depth image arrives, if we have a valid banana center from YOLO,
        we align depth to the color frame, read the depth at that pixel, and compute
        the 3D coords in the base frame. Finally, publish a PoseStamped.
        """
        if self.center_coordinates is None or self.center_coordinates == (None, None) or self.rgb is None:
            # No valid banana center or no recent RGB
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            aligned_depth = self.align_depth(depth, self.depth_K, self.rgb, self.rgb_K, self.cam2cam_transform)

            # Depth is in millimeters at banana pixel
            x_pix, y_pix = self.center_coordinates
            depth_at_center_mm = aligned_depth[y_pix, x_pix]
            depth_at_center_m = depth_at_center_mm / 1000.0
            self.get_logger().info(f"Depth at banana center = {depth_at_center_m:.3f} m")

            # Convert pixel coords + depth to camera coordinates
            camera_coords = self.pixel_to_camera_frame(self.center_coordinates, depth_at_center_m)
            # Transform camera coords to the robot base frame
            base_coords = self.camera_to_base_tf(camera_coords)
            self.get_logger().info(f"Banana in base frame: {base_coords}")

            if base_coords is not None:
                # Publish as a PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "map"  # or "locobot/arm_base_link", etc.

                # base_coords is shape (4,1) => [x, y, z, 1]
                pose_msg.pose.position.x = float(base_coords[0])
                pose_msg.pose.position.y = float(base_coords[1])
                pose_msg.pose.position.z = float(base_coords[2])
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

                self.publish_target.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {str(e)}")

    # ---------------------------------------------------------------------
    #  YOLO Inference + Creating Two Annotated Images
    # ---------------------------------------------------------------------
    def run_yolo_and_get_images(self, frame: np.ndarray, model_path: str = "yolov8n.pt"):
        """
        Runs YOLO on the given frame and creates two images:
         1) 'all_detections_frame': bounding boxes for *all* detections (any class).
         2) 'banana_frame': bounding boxes only for banana(s), if any. If no banana, None.

        Returns (all_detections_frame, banana_frame, banana_center).
          - all_detections_frame: annotated with every detection
          - banana_frame: annotated only with banana detection(s), or None if no bananas
          - banana_center: (x_center, y_center) of the highest-confidence banana,
                           or (None, None) if no banana

        The user can then save these images as separate files.
        """
        model = YOLO(model_path)
        results = model.predict(frame)
        detections = results[0]  # Single image => single Results

        # We'll annotate on separate copies to keep logic simple
        all_detections_frame = frame.copy()
        banana_frame = None

        best_banana_conf = 0.0
        banana_center = (None, None)

        # We'll track *all* banana boxes so we can create a banana-only image if needed
        banana_boxes = []

        # 1) Annotate all detections on 'all_detections_frame'
        for box in detections.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = detections.names[cls_id]

            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Draw rectangle for this detection
            cv2.rectangle(
                all_detections_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # color
                2             # thickness
            )
            # Label text
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                all_detections_frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            # 2) If it's a banana, track in banana_boxes
            if class_name.lower() == "banana":
                banana_boxes.append((x1, y1, x2, y2, conf))

                # If needed, keep track of best (highest confidence) banana for center
                if conf > best_banana_conf:
                    best_banana_conf = conf
                    bx_center = int((x1 + x2) / 2)
                    by_center = int((y1 + y2) / 2)
                    banana_center = (bx_center, by_center)

        # 3) If we found any bananas, create a 'banana_frame' with ONLY banana boxes
        if len(banana_boxes) > 0:
            banana_frame = frame.copy()
            for (x1, y1, x2, y2, conf) in banana_boxes:
                # Draw rectangle for bananas only
                cv2.rectangle(
                    banana_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )
                label = f"banana {conf:.2f}"
                cv2.putText(
                    banana_frame,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return all_detections_frame, banana_frame, banana_center

    # ---------------------------------------------------------------------
    # COORDINATE TRANSFORMS, ALIGNMENTS, ETC.
    # ---------------------------------------------------------------------
    def pixel_to_camera_frame(self, pixel_coords, depth_m):
        """
        Convert pixel coordinates + depth (meters) to camera coordinates.
        """
        fx, fy, cx, cy = self.rgb_K  # (fx, fy, cx, cy)
        u, v = pixel_coords
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m
        return (X, Y, Z)

    def camera_to_base_tf(self, camera_coords):
        """
        Use TF to transform from 'camera_color_optical_frame' to 'locobot/arm_base_link'.
        Returns a 4x1 array [x, y, z, 1] in base frame, or None on error.
        """
        try:
            if self.tf_buffer.can_transform('locobot/arm_base_link',
                                            'camera_color_optical_frame',
                                            rclpy.time.Time()):
                transform_camera_to_base = self.tf_buffer.lookup_transform(
                    'locobot/arm_base_link',
                    'camera_color_optical_frame',
                    rclpy.time.Time())

                tf_geom = transform_camera_to_base.transform

                trans = np.array([tf_geom.translation.x,
                                  tf_geom.translation.y,
                                  tf_geom.translation.z], dtype=float)
                rot = np.array([tf_geom.rotation.x,
                                tf_geom.rotation.y,
                                tf_geom.rotation.z,
                                tf_geom.rotation.w], dtype=float)

                transform_mat = self.create_transformation_matrix(rot, trans)
                camera_coords_homogenous = np.array([[camera_coords[0]],
                                                     [camera_coords[1]],
                                                     [camera_coords[2]],
                                                     [1]])
                base_coords = transform_mat @ camera_coords_homogenous
                return base_coords
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to convert camera->base transform: {str(e)}")
            return None

    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """ Create a 4x4 homogeneous transform from (x, y, z, w) quaternion and (tx, ty, tz). """
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def align_depth(self, depth: np.ndarray,
                    depth_K: Tuple[float,float,float,float],
                    rgb: np.ndarray,
                    rgb_K: Tuple[float,float,float,float],
                    cam2cam_transform: np.ndarray) -> np.ndarray:
        """
        Align depth image to the rgb image for consistent (u, v) indexing.
        """
        old_fx, old_fy, old_cx, old_cy = depth_K
        new_fx, new_fy, new_cx, new_cy = rgb_K

        K_old = np.array([[old_fx, 0, old_cx],
                          [0, old_fy, old_cy],
                          [0,     0,     1]])
        K_new = np.array([[new_fx, 0, new_cx],
                          [0, new_fy, new_cy],
                          [0,     0,     1]])

        # Step 1: rescale depth intrinsics to match RGB resolution
        depth_rescaled = self.convert_intrinsics(depth, K_old, K_new,
                                                 new_size=(rgb.shape[1], rgb.shape[0]))

        # Step 2: warp perspective from depth camera to RGB camera transform
        R_ = cam2cam_transform[:3, :3]
        t_ = cam2cam_transform[:3, 3]
        aligned_depth = self.warp_image(depth_rescaled, K_new, R_, t_)

        return aligned_depth

    def convert_intrinsics(self, img, K_old, K_new, new_size=(1280, 720)):
        """
        Converts one image to new intrinsics by building a map of old->new coords.
        """
        width, height = new_size
        K_new_inv = np.linalg.inv(K_new)

        # Generate pixel grid in new image coords
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        homogenous_coords = np.stack([
            x.ravel(),
            y.ravel(),
            np.ones_like(x).ravel()
        ], axis=-1).T

        # Map from new coords -> old coords
        old_coords = K_old @ (K_new_inv @ homogenous_coords)
        old_coords /= old_coords[2, :]  # Normalize

        map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
        map_y = old_coords[1, :].reshape(height, width).astype(np.float32)

        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def warp_image(self, image, K, R, t):
        """
        Warps image from camera1 to camera2 using a homography approximation.
        For real 3D transforms, you'd do a full reprojection.
        """
        H = self.compute_homography(K, R, t)
        height, width = image.shape[:2]
        return cv2.warpPerspective(image, H, (width, height))

    def compute_homography(self, K, R, t):
        """
        Creates a homography ignoring real depth variation. 
        Approx: H = K * R * K^-1
        """
        K_inv = np.linalg.inv(K)
        return K @ (R @ K_inv)

def main(args=None):
    rclpy.init(args=args)
    camera = Camera()
    rclpy.spin(camera)
    camera.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
