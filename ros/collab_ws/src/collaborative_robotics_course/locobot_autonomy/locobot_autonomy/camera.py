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

        self.detector = VisionObjectDetector()

        self.saved_image = False

        # self.mobile_base_vel_publisher = self.create_publisher(Twist,"/locobot/mobile_base/cmd_vel", 1)

        # msg = Twist()
        # msg.angular.z = 0.5  # Set angular velocity (turn)
        # self.mobile_base_vel_publisher.publish(msg)

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

    def camera_listener_callback(self, msg):
        self.get_logger().info('Received an image')

        if self.saved_image is False:

            if msg is None:
                self.get_logger().error("Received an empty image message!")
                return

            try:
                cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # May need to convert image to bytes first with
                success, encoded_image = cv2.imencode('.jpg', cv_ColorImage)
                image_bytes = encoded_image.tobytes()

                # import pdb; pdb.set_trace()
                    
                center_coordinates, vertices = self.detector.find_center_vertices(image_bytes, 'Banana')
                self.get_logger().info(f"Center Coords: {center_coordinates}")
                annotated_frame = self.detector.annotate_image(vertices, cv_ColorImage)
                # self.get_logger().info(f"annotated_frame: {annotated_frame}")
                annotated_frame_np = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
                
                cv2.imwrite("/home/locobot/team_chinese_rohan_ws/ME326FinalProject/ros/collab_ws/src/collaborative_robotics_course/locobot_autonomy/locobot_autonomy/annotated_fruit.jpg", annotated_frame)
                self.saved_image = True

                # print("Center in pixel coordinates: ", center_coordinates)

                # cv2.imshow('Camera Stream', annotated_frame_np)
                # cv2.waitKey(1)

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