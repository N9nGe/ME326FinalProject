#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge
import os
from google.cloud import vision
from PIL import Image as PILImage, ImageDraw, ImageFont
import io
import time

from align_depth_fncs import align_depth # This import may need to be adjusted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import rclpy
from rclpy.node import Node
import sensor_msgs
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import geometry_msgs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from rclpy.qos import qos_profile_sensor_data, QoSProfile

# Frame Imports
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

OPENING_GRIPPER = 4
POSTIONING_EE = 5
CLOSING_GRIPPER = 6
RAISING_EE = 7

class GrabNode(Node):
    def __init__(self):
        super().__init__("grab_node")

        ## Google Vision Initialization
        self.json_key_path = "/home/locobot/Downloads/winged-complex-448806-d7-1c422cf64607.json"
        #self.json_key_path = "/home/ubuntu/Desktop/collaborative/collaborative-robotics/collaborative-robotics-9c1946cd9dbc.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.json_key_path

        import pdb; pdb.set_trace()
        self.client = vision.ImageAnnotatorClient()

        ## CV Bridge Initialization
        self.bridge = CvBridge()

        ## Desired object to grab
        self.desiredObject = "Shoe"

        ## Initialize state of robot
        self.subState = OPENING_GRIPPER

        # Frame Listener Initialization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        target_frame = 'locobot/arm_base_link'
        source_frame = 'camera_locobot_link'
        now = rclpy.time.Time()

        self.timer = self.create_timer(3.0, self.test)

        return


        transform_stamped = self.GetTransform(target_frame, source_frame)
        
        translate2Base = None
        rotate2Base_quat = None
        if transform_stamped:
            translate2Base = transform_stamped.transform.translation
            rotate2Base_quat = transform_stamped.transform.rotation
            #self.get_logger().info(f'Transform from camera_locobot_link to locobot_arm_base_link:')
            #self.get_logger().info(f'Translation: x={translation.x:.2f}, y={translation.y:.2f}, z={translation.z:.2f}')
            #self.get_logger().info(f'Rotation: x={rotation.x:.2f}, y={rotation.y:.2f}, z={rotation.z:.2f}, w={rotation.w:.2f}')
        else:
            self.get_logger().warn(f'Failed to lookup transform from camera_locobot_link to locobot_arm_base_link')
        baseRotationObject = R.from_quat([rotate2Base_quat.x,rotate2Base_quat.y,rotate2Base_quat.z,rotate2Base_quat.w])
        baseRotMat = baseRotationObject.as_matrix()
        self.baseTransform = np.zeros((3,4))
        self.baseTransform[:,:3] = baseRotMat
        self.baseTransform[0,3] = translate2Base.x
        self.baseTransform[1,3] = translate2Base.y
        self.baseTransform[2,3] = translate2Base.z

        ## Publishers for moving different parts of robot
        self.mobile_base_vel_publisher = self.create_publisher(Twist, '/locobot/mobile_base/tf', 10)
        self.arm_publisher = self.create_publisher(PoseStamped,'/arm_pose',10)
        self.gripper_publisher = self.create_publisher(Bool,'/gripper',10)

        # Camera Intrinsic Subscribers
        self.CameraIntrinicsSubscriber = self.create_subscription(CameraInfo,"/locobot/camera/color/camera_info",self.GetCameraIntrinsics,1)
        self.DepthCameraIntrinicsSubscriber = self.create_subscription(CameraInfo,"/locobot/camera/depth/camera_info",self.GetDepthCameraIntrinsics,1)

        # Depth Camera Subscriber
        self.depth_camera_subscription = self.create_subscription(Image,"/locobot/camera/depth/image_rect_raw",self.GetDepthCV2Image,qos_profile=qos_profile_sensor_data)
        self.depth_camera_subscription  # prevent unused variable warning

        # RGB Camera Subscriber
        self.camera_subscription = self.create_subscription(Image,"/locobot/camera/color/image_raw",self.LocateDesiredObject,qos_profile=qos_profile_sensor_data)
        self.camera_subscription  # prevent unused variable warning

        ## Variable Initializations
        self.cv_DepthImage = None
        self.alpha = None
        self.beta = None
        self.u0 = None
        self.v0 = None
        self.alpha_depth = None
        self.beta_depth = None
        self.u0_depth = None
        self.v0_depth = None
        self.haveIntrinsics_RBG = False
        self.haveIntrinsics_Depth = False
        self.haveDepthImage = False

        ## Ensure the robot is stationary (I recall Zhongchun saying something about how the base will move at the start to orient - this is too chaotic right now)
        msg = Twist()
        self.mobile_base_vel_publisher.publish(msg)


    def test(self):
        # import pdb; pdb.set_trace()
        target_frame = 'locobot/arm_base_link'
        source_frame = 'camera_locobot_link'
        now = rclpy.time.Time()
        transform_stamped = self.GetTransform(target_frame, source_frame)
    
    def GetCameraIntrinsics(self, CameraMsg):
        self.alpha = CameraMsg.k[0]
        self.beta = CameraMsg.k[4]
        self.u0 = CameraMsg.k[2]
        self.v0 = CameraMsg.k[5]
        self.destroy_subscription(self.CameraIntrinicsSubscriber)
        self.haveIntrinsics_RBG = True
    
    def GetDepthCameraIntrinsics(self, DepthCameraMsg):
        self.alpha_depth = DepthCameraMsg.k[0]
        self.beta_depth = DepthCameraMsg.k[4]
        self.u0_depth = DepthCameraMsg.k[2]
        self.v0_depth = DepthCameraMsg.k[5]
        self.destroy_subscription(self.DepthCameraIntrinicsSubscriber)
        self.haveIntrinsics_Depth = True

    def GetDepthCV2Image(self,DepthImageMsg):
        self.cv_DepthImage = self.bridge.imgmsg_to_cv2(DepthImageMsg, desired_encoding='passthrough')
        self.haveDepthImage = True
    
    def GetTransform(self, target_frame, source_frame):
        try:
            now = self.get_clock().now()
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0)
            )
            self.get_logger().info(f'{transform}')
            return transform
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None
    
    def GetObjectInBaseFrame(self,desiredObjectAttributes,cv_image,image_bytes):
        bounding_box = desiredObjectAttributes.bounding_poly
        # Find the center from the corners of the bounding box
        x_min = None
        y_min = None
        x_max = None
        y_max = None
        for vertex in bounding_box.normalized_vertices:
            if x_min != None:
                if vertex.x < x_min:
                    x_min = vertex.x
                if vertex.x > x_max:
                    x_max = vertex.x
                if vertex.y < y_min:
                    y_min = vertex.y
                if vertex.y > y_max:
                    y_max = vertex.y
            else:
                x_min = vertex.x
                x_max = vertex.x
                y_min = vertex.y
                y_max = vertex.y
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        # Return the center in pixel coordinates.
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
        pixel_x = center_x * width
        pixel_y = center_y * height

        # Matt's Functionality!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        rgb_intriniscs = (self.alpha,self.beta,self.u0,self.v0)
        depth_intrinsics = (self.alpha_depth,self.beta_depth,self.u0_depth,self.v0_depth)
        aligned_depth = align_depth(self.cv_DepthImage,depth_intrinsics,cv_image,rgb_intriniscs,np.eye(4))
        # Double check that the x and y pixels are what is used (in this ordering) to accept depth element!!!!!!!!!!!!!!!!!!!!!!
        Z_c = aligned_depth[pixel_y,pixel_x]
        X_c = (pixel_x - self.u0)*Z_c/self.alpha
        Y_c = (pixel_y - self.v0)*Z_c/self.beta
        point_c_H = np.array([X_c,Y_c,Z_c,1])
        point_base = self.baseTransform @ point_c_H
        return point_base
    
    def LocateDesiredObject(self,imageMessage):
        if self.haveDepthImage and self.haveIntrinsics_Depth and self.haveIntrinsics_RBG:
            # Convert Image Message to CV Image
            cv_ColorImage = self.bridge.imgmsg_to_cv2(imageMessage, desired_encoding='passthrough')
            # For Google Vision to process, the image must be expressed in bytes
            success, encoded_image = cv2.imencode('.jpg', cv_ColorImage)
            image_bytes = encoded_image.tobytes()
            visionImage = vision.Image(content=image_bytes)
            # Send the image to the API for object localization
            response = self.client.object_localization(image=visionImage)
            # Extract localized object annotations
            objects = response.localized_object_annotations
            foundDesiredObject = False
            desiredObjectAttributes = None
            for object in objects:
                if object.name.lower() == self.desiredObject:
                    foundDesiredObject = True
                    desiredObjectAttributes = object
                    break
            
            if foundDesiredObject:
                self.point_base = self.GetObjectInBaseFrame(desiredObjectAttributes,cv_ColorImage,image_bytes)
                self.get_logger().info(f'Desired object located at: x={self.point_base[0]:.2f}, y={self.point_base[1]:.2f}, z={self.point_base[2]:.2f}')
                self.destroy_subscription(self.camera_subscription)
                self.destroy_subscription(self.depth_camera_subscription)
                self.timer = self.create_timer(3.0, self.Grab_StateMachine)
    
    def Grab_StateMachine(self):
        # Since this code won't be blocked as the arm/gripper move, we need to manually introduce delays between each gripper/arm command
        if self.subState == OPENING_GRIPPER:
            self.get_logger().info(f'Opening Gripper')
            msgOpenGripper = Bool()
            msgOpenGripper.data = True
            self.gripper_publisher.publish(msgOpenGripper) # open gripper
            self.subState = POSTIONING_EE
        elif self.subState == POSTIONING_EE:
            self.get_logger().info(f'Positioning End Effector')
            EE_rotationObject = R.from_rotvec(np.pi/2 * np.array([0,1,0]))
            EE_rotation_quat = EE_rotationObject.as_quat()
            msgEE = PoseStamped()
            msgEE.pose.position.x = self.point_base[0]
            msgEE.pose.position.y = self.point_base[1]
            msgEE.pose.position.z = self.point_base[2]
            msgEE.pose.orientation.x = EE_rotation_quat[0]
            msgEE.pose.orientation.y = EE_rotation_quat[1]
            msgEE.pose.orientation.z = EE_rotation_quat[2]
            msgEE.pose.orientation.w = EE_rotation_quat[3]
            self.arm_publisher.publish(msgEE) # place end effector at location of object
            self.subState = CLOSING_GRIPPER
        elif self.subState == CLOSING_GRIPPER:
            self.get_logger().info(f'Closing Gripper')
            msgCloseGripper = Bool()
            msgCloseGripper.data = False
            self.gripper_publisher.publish(msgCloseGripper) # close gripper
            self.subState = RAISING_EE
        elif self.subState == RAISING_EE:
            self.get_logger().info(f'Raising End Effector')
            point_lift = self.point_base + np.array([0.2,0,0.16])
            EE_lift_rotationObject = R.from_rotvec(np.array([0,0,0]))
            EE_lift_rotation_quat = EE_lift_rotationObject.as_quat()
            msgEE_lift = PoseStamped()
            msgEE_lift.pose.position.x = point_lift[0]
            msgEE_lift.pose.position.y = point_lift[1]
            msgEE_lift.pose.position.z = point_lift[2]
            msgEE_lift.pose.orientation.x = EE_lift_rotation_quat[0]
            msgEE_lift.pose.orientation.y = EE_lift_rotation_quat[1]
            msgEE_lift.pose.orientation.z = EE_lift_rotation_quat[2]
            msgEE_lift.pose.orientation.w = EE_lift_rotation_quat[3]
            self.arm_publisher.publish(msgEE_lift) # raise object
            self.timer.cancel()

if __name__ == '__main__':
    rclpy.init()
    node = GrabNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()