#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
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
from nav_msgs.msg import Odometry

json_key_path = "/home/ubuntu/Desktop/collaborative/colab_ws/collaborative-robotics/ros/collab_ws/me326-hw2-02b887b1a25c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

class Camera(Node):
    def __init__(self):
        super().__init__('camera')
        self.bridge = CvBridge()

        # Set QoS to Reliable, matching most camera publishers
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self.subscription = self.create_subscription(
            Image,
            '/locobot/camera_frame_sensor/image_raw',
            self.listener_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

        self.detector = VisionObjectDetector()

        self.mobile_base_vel_publisher = self.create_publisher(Twist,"/locobot/mobile_base/cmd_vel", 1)

        msg = Twist()
        msg.angular.z = 0.5  # Set angular velocity (turn)
        self.mobile_base_vel_publisher.publish(msg)

    def convert_cv2_to_bytes(cv_image, format="JPEG"):
        # Convert from BGR (OpenCV format) to RGB (PIL format)
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image
        pil_image = PILImage.fromarray(image_rgb)
        
        # Save as bytes
        image_bytes_io = io.BytesIO()
        pil_image.save(image_bytes_io, format=format)  # Format can be "JPEG" or "PNG"
        
        # Get the byte data
        image_bytes = image_bytes_io.getvalue()
        
        return image_bytes

    def listener_callback(self, msg):
        self.get_logger().info('Received an image')

        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            image_bytes = Camera.convert_cv2_to_bytes(cv_image)
            
            center_coordinates = self.detector.find_center(image_bytes, 'blue')
            annotated_frame = self.detector.annotate_image(image_bytes)
            annotated_frame_np = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

            print("Center in pixel coordinates: ", center_coordinates)

            cv2.imshow('Camera Stream', annotated_frame_np)
            cv2.waitKey(1)

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