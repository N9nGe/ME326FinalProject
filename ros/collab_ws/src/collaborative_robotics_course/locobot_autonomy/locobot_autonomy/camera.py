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
from vision_object_detector import VisionObjectDetector
import io
from PIL import Image as PILImage

import geometry_msgs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
from rclpy.duration import Duration

json_key_path = "/home/locobot/Naixiang/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/locobot_autonomy/me326-hw2-02b887b1a25c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path                                   

class Camera(Node):
    def __init__(self):
        super().__init__('rohan')
        self.bridge = CvBridge()

        # Set QoS to Reliable, matching most camera publishers
        # qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        self.subscription = self.create_subscription(
            Image,
            '/locobot/camera/color/image_raw',
            self.camera_listener_callback, 10)
        self.subscription  # prevent unused variable warning

        self.subscription_depth = self.create_subscription(
            Image,
            '/locobot/camera/depth/image_rect_raw',
            self.depth_listener_callback, 10)
        self.subscription_depth  # prevent unused variable warning

        self.subscription_audio = self.create_subscription( 
            String,
            'AudioItem', # CHANGE THIS TO THE CORRECT TOPIC
            self.audio_listener_callback, 10)
        self.subscription_audio  # prevent unused variable warning

        self.publish_target = self.create_publisher(PoseStamped, '/arm_pose', 10) # CHANGE THIS TO THE CORRECT TOPIC
        self.publish_target  # prevent unused variable warning

        self.detector = VisionObjectDetector()

        self.rgb_K = (609.03759765625, 609.2069091796875, 323.6105041503906, 243.78759765625)
        self.depth_K = (379.8849792480469, 379.8849792480469, 320.3520202636719, 379.8849792480469)
        self.cam2cam_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.center_coordinates = None
        self.object = None
        self.rgb = None
        self.rgb_width = 640
        self.rgb_height = 480

        # PUBLISHER FOR MOBILE BASE
        # self.mobile_base_vel_publisher = self.create_publisher(Twist,"/locobot/mobile_base/cmd_vel", 1)
        # msg = Twist()
        # msg.angular.z = 0.5  # Set angular velocity (turn)
        # self.mobile_base_vel_publisher.publish(msg)

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def pixel_to_world(self, pixel_coords, depth):
        xc = pixel_coords[0] * depth / self.rgb_K[1]
        yc = pixel_coords[1] * depth / self.rgb_K[2]
        return (xc, yc, depth)
    
    def camera_to_base_tf(self, camera_coords):
        try:
            # Check if both transforms are available
            if self.tf_buffer.can_transform('locobot/arm_base_link', 'camera_color_frame', rclpy.time.Time()):
                # Get the transformation from camera to base
                transform_camera_to_base = self.tf_buffer.lookup_transform('locobot/arm_base_link', 'camera_color_frame', rclpy.time.Time())

                tf_geom = transform_camera_to_base.transform

                # Get translation
                trans = np.array([tf_geom.translation.x,
                                tf_geom.translation.y,
                                tf_geom.translation.z],
                                dtype=float)

                # Get rotation quaternion [x, y, z, w]
                rot = np.array([tf_geom.rotation.x,
                                tf_geom.rotation.y,
                                tf_geom.rotation.z,
                                tf_geom.rotation.w],
                            dtype=float)
                
                transformation_matrix = self.create_transformation_matrix(rot, trans)
                camera_coords_homogenous = np.array([[camera_coords[0]], [camera_coords[1]], [camera_coords[2]], [1]])
                world_coords_m = (transformation_matrix @ camera_coords_homogenous) / 1000
                self.get_logger().info(f'World Coordinates in meters: {world_coords_m}')
                return world_coords_m
      
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to convert depth image: {str(e)}")

    def audio_listener_callback(self, msg):
        self.object = msg.data

    def camera_listener_callback(self, msg):
        self.get_logger().info('Received an image')

        # if self.saved_image is False:

        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return

        try:
            cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # self.get_logger().info(f'cv_COLORIMAGE SHAPE {cv_ColorImage.shape}')
            success, encoded_image = cv2.imencode('.jpg', cv_ColorImage)
            image_bytes = encoded_image.tobytes()

            # import pdb; pdb.set_trace()
                
            self.center_coordinates, vertices = self.detector.find_center_vertices(image_bytes, "banana") # REPLACE WITH OBJECT NAME
            # self.get_logger().info(f"Center Coords: {self.center_coordinates}")

            # if self.center_coordinates is not None:
            #     msg = Twist()
            self.rgb = cv_ColorImage
                # self.mobile_base_vel_publisher.publish(msg) # PUBLISH TO MOBILE BASE

            annotated_frame = self.detector.annotate_image(vertices, cv_ColorImage)
            # self.get_logger().info(f"annotated_frame: {annotated_frame}")
            # annotated_frame_np = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            
            # cv2.imwrite("/home/locobot/team_chinese_rohan_ws/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/locobot_autonomy/annotated_fruit.jpg", annotated_frame)
            self.saved_image = True

                # print("Center in pixel coordinates: ", center_coordinates)

                # cv2.imshow('Camera Stream', annotated_frame_np)
                # cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {str(e)}")

    def depth_listener_callback(self, msg):

            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                aligned_depth = self.align_depth(depth, self.depth_K, self.rgb, self.rgb_K, self.cam2cam_transform)
                depth_at_center = aligned_depth[self.center_coordinates[1], self.center_coordinates[0]]
                center_rgb = list(self.center_coordinates)
                center_rgb[0] = self.center_coordinates[0] - self.rgb_width/2
                center_rgb[1] = self.center_coordinates[1] - self.rgb_height/2
                world_coords = self.pixel_to_world(center_rgb, depth_at_center)
                world_coords_base = self.camera_to_base_tf(world_coords)
                self.get_logger().info(f"World Coords in Base Frame: {world_coords_base}")
                

                msg = PoseStamped()

                msg.header.stamp.sec = 1710000000  # Replace with actual time or use a Clock
                msg.header.stamp.nanosec = 0
                msg.header.frame_id = "map"

                msg.pose.position.x = float(world_coords_base[0])
                msg.pose.position.y = float(world_coords_base[1])
                msg.pose.position.z = float(world_coords_base[2])

                msg.pose.orientation.x = 0.0
                msg.pose.orientation.y = 0.0
                msg.pose.orientation.z = 0.0
                msg.pose.orientation.w = 1.0

                self.publish_target.publish(msg)
                
            except Exception as e:
                self.get_logger().error(f"Failed to convert depth image: {str(e)}")

    def convert_intrinsics(self, img, K_old, K_new, new_size=(1280, 720)):
        """
        Convert a set of images to a different set of camera intrinsics.
        Parameters:
        - images: List of input images.
        - K_old: Matrix of the old camera intrinsics.
        - K_new: Matrix of the new camera intrinsics.
        - new_size: Tuple (width, height) defining the size of the output images.
        Returns:
        - List of images converted to the new camera intrinsics.
        """
        width, height = new_size
        # self.get_logger().info("in convert intrinsics")
        # Compute the inverse of the new intrinsics matrix for remapping
        K_new_inv = np.linalg.inv(K_new)
        # self.get_logger().info(f"K_new_inv: ")
        # Construct a grid of points representing the new image coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        homogenous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T
        # self.get_logger().info("homogenous_coords: ", homogenous_coords)
        # Convert to the old image coordinates
        old_coords = K_old @ K_new_inv @ homogenous_coords
        old_coords /= old_coords[2, :]  # Normalize to make homogeneous
        # self.get_logger.info("old coords: ", old_coords)
        # Reshape for remapping
        map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
        map_y = old_coords[1, :].reshape(height, width).astype(np.float32)
        
        # Remap the image to the new intrinsics
        converted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return converted_img

    def warp_image(self, image, K, R, t):
        """
        Warp an image from the perspective of camera 1 to camera 2.

        :param image: Input image from camera 1
        :param K: Intrinsic matrix of both cameras
        :param R: Rotation matrix from camera 1 to camera 2
        :param t: Translation vector from camera 1 to camera 2
        :return: Warped image as seen from camera 2
        """
        # Compute the homography matrix
        H = self.compute_homography(K, R, t)

        # Warp the image using the homography
        height, width = image.shape[:2]
        warped_image = cv2.warpPerspective(image, H, (width, height))

        return warped_image


    def compute_homography(self, K, R, t):
        """
        Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
        """
        K_inv = np.linalg.inv(K)
        H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
        return H


    def align_depth(self, depth: np.ndarray, depth_K: Tuple[float,float,float,float], rgb: np.ndarray, rgb_K: Tuple[float,float,float,float], cam2cam_transform: np.ndarray) -> np.ndarray:
        """
        align depth image to the rgb image.

        :param depth: depth image
        :param depth_K: intrinsics of the depth camera
        :param rgb: rgb image
        :param rgb_K: intrinsics of the rgb camera
        :param cam2cam_transform: transformation matrix from depth to rgb camera
        :return: aligned depth image
        """
        
        old_fx, old_fy, old_cx, old_cy = depth_K
        new_fx, new_fy, new_cx, new_cy = rgb_K
        # Constructing the old and new intrinsics matrices
        K_old = np.array([[old_fx, 0, old_cx], [0, old_fy, old_cy], [0, 0, 1]])
        K_new = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])

        # conver the instrinsics of the depth camera to the intrinsics of the rgb camera.
        depth = self.convert_intrinsics(depth, K_old, K_new, new_size=(rgb.shape[1], rgb.shape[0]))
        # warp the depth image to the rgb image with the transformation matrix from camera to camera.
        depth = self.warp_image(depth, K_new, cam2cam_transform[:3, :3], cam2cam_transform[:3, 3])        
        return depth
    
    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Create a 4x4 homogeneous transformation matrix from a quaternion and translation vector.

        Parameters:
        - quaternion: np.ndarray of shape (4,) [x, y, z, w]
        - translation: np.ndarray of shape (3,) [x, y, z]

        Returns:
        - 4x4 homogeneous transformation matrix
        """
        # Convert quaternion (x, y, z, w) to 3x3 rotation matrix
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Create 4x4 homogeneous transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation

        return matrix
    
    def decompose_transformation_matrix(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose a 4x4 homogeneous transformation matrix into a quaternion and translation vector.

        Parameters:
        - matrix: np.ndarray of shape (4, 4)

        Returns:
        - quaternion: np.ndarray of shape (4,) [x, y, z, w]
        - translation: np.ndarray of shape (3,) [x, y, z]
        """
        # Extract rotation matrix and translation vector
        rotation_matrix = matrix[:3, :3]
        translation = matrix[:3, 3]

        # Convert rotation matrix to quaternion (x, y, z, w)
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        return quaternion, translation

def main(args=None):
    rclpy.init(args=args)
    camera = Camera()
    rclpy.spin(camera)
    camera.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()