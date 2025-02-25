#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import cv2

import os
import numpy as np
from vision_object_detector import VisionObjectDetector
import io
from PIL import Image as PILImage
from align_depth_fncs import convert_intrinsics, warp_image, compute_homography, align_depth

import geometry_msgs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

json_key_path = "/home/locobot/team_chinese_rohan_ws/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/locobot_autonomy/me326-hw2-02b887b1a25c.json"
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
            '/locobot/camera/depth/image_raw',
            self.depth_listener_callback, 10)
        self.subscription_depth  # prevent unused variable warning

        self.subscription_audio = self.create_subscription( 
            String,
            '/locobot/speech', # CHANGE THIS TO THE CORRECT TOPIC
            self.audio_listener_callback, 10)
        self.subscription_audio  # prevent unused variable warning

        self.publish_target = self.create_publisher(Float32MultiArray, '/locobot/move_base_simple/goal', 10) # CHANGE THIS TO THE CORRECT TOPIC
        self.publish_target  # prevent unused variable warning

        self.detector = VisionObjectDetector()

        self.saved_image = False

        focal = 0.1
        kx = 1
        ky = 1
        self.alpha = kx * focal
        self.beta = ky * focal

        self.rgb_K = (kx * focal, ky * focal, 0, 0)
        self.depth_K = (360.01, 360.01, 243.87, 137.92)
        self.cam2cam_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.center_coordinates = None
        self.object = None
        self.rgb = None

        # PUBLISHER FOR MOBILE BASE
        # self.mobile_base_vel_publisher = self.create_publisher(Twist,"/locobot/mobile_base/cmd_vel", 1)
        # msg = Twist()
        # msg.angular.z = 0.5  # Set angular velocity (turn)
        # self.mobile_base_vel_publisher.publish(msg)

    def pixel_to_world(self, pixel_coords, depth):
        xc = pixel_coords[0] * depth / self.alpha
        yc = pixel_coords[1] * depth / self.beta
        return (xc, yc, depth)

    def audio_listener_callback(self, msg):
        self.object = msg.data

    def camera_listener_callback(self, msg):
        self.get_logger().info('Received an image')

        if self.saved_image is False:

            if msg is None:
                self.get_logger().error("Received an empty image message!")
                return

            try:
                cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                success, encoded_image = cv2.imencode('.jpg', cv_ColorImage)
                image_bytes = encoded_image.tobytes()

                # import pdb; pdb.set_trace()
                    
                self.center_coordinates, vertices = self.detector.find_center_vertices(image_bytes, 'Banana') # REPLACE WITH OBJECT NAME
                self.get_logger().info(f"Center Coords: {self.center_coordinates}")

                if self.center_coordinates is not None:
                    msg = Twist()
                    self.rgb = cv_ColorImage
                    # self.mobile_base_vel_publisher.publish(msg) # PUBLISH TO MOBILE BASE

                annotated_frame = self.detector.annotate_image(vertices, cv_ColorImage)
                # self.get_logger().info(f"annotated_frame: {annotated_frame}")
                # annotated_frame_np = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
                
                cv2.imwrite("/home/locobot/team_chinese_rohan_ws/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/locobot_autonomy/annotated_fruit.jpg", annotated_frame)
                self.saved_image = True

                # print("Center in pixel coordinates: ", center_coordinates)

                # cv2.imshow('Camera Stream', annotated_frame_np)
                # cv2.waitKey(1)

            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {str(e)}")

    def depth_listener_callback(self, msg):
        if self.center_coordinates is not None:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                aligned_depth = align_depth(depth, self.depth_K, self.rgb, self.rgb_K, self.cam2cam_transform)
                depth_at_center = aligned_depth[self.center_coordinates[0], self.center_coordinates[1]]
                self.get_logger().info(f"Depth at center: {depth_at_center}")

                world_coords = self.pixel_to_world(self.center_coordinates, depth_at_center)
                self.get_logger().info(f"World Coords: {world_coords}")

                msg = Float32MultiArray()
                msg.data = np.array(world_coords)
                self.publish_target.publish(msg)
                
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    camera = Camera()
    rclpy.spin(camera)
    camera.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()