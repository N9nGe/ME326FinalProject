#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS
from locobot_wrapper_msgs.action import MoveArm, MoveGripper
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation as R
import numpy as np
from std_msgs.msg import Bool
import time

class ArmWrapperNode(Node):
    def __init__(self):
        super().__init__('arm_wrapper_node')
        # Declare the parameter 'use_sim' with a default value of False
        self.declare_parameter('use_sim', False)
        
        # Get the parameter value
        self.use_sim = self.get_parameter('use_sim').get_parameter_value().bool_value
        
        # Log the value of 'use_sim'
        self.get_logger().info(f"use_sim parameter set to: {self.use_sim}")

        self.received_pose = False

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/arm_pose',  
            self.pose_callback,
            10
        )

        self.gripper_subscriber = self.create_subscription(
            Bool,
            '/gripper',
            self.gripper_callback,
            10
        )

        self._action_client = ActionClient(
            self,
            MoveArm,
            'movearm'
        )

        self.__gripper_client = ActionClient(
            self,
            MoveGripper,
            'movegripper'
        )

        if not self.use_sim:
            self.locobot = InterbotixLocobotXS(
            robot_model='locobot_wx250s',
            robot_name='locobot',
            arm_model='mobile_wx250s'
            )

    def pose_callback(self, msg):
        # Log the received PoseStamped message

        if self.received_pose:
            return  # Ignore further messages

        self.received_pose = True  # Set the flag to stop updates
        self.get_logger().info(f"Received Pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        
        # Extract the target pose
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        target_z = msg.pose.position.z

        hover_height = 0.1  # Height above the object for hover
        lift_height = 0.3  # Height to lift after grasping

        # Step 1: Move above the target position (hover)
        time.sleep(3.0)
        self.move_to_pose(target_x, target_y, target_z + hover_height)
        self.get_logger().info("Step 1: Moved above the target position (hover).")
        time.sleep(3.0)
        

        # Step 2: Open the gripper
        self.control_gripper(True)
        self.get_logger().info("Step 2: Gripper opened.")
        time.sleep(3.0)
        

        # Step 3: Move down to the target position
        self.move_to_pose(target_x , target_y, target_z - 0.05 )
        self.get_logger().info("Step 3: Moved to target position.")
        time.sleep(3.0)
        

        # Step 4: Close the gripper
        self.control_gripper(False)
        self.get_logger().info("Step 4: Gripper closed (object grasped).")
        time.sleep(3.0)
        

        # Step 5: Lift the object back to hover height
        self.move_to_pose(target_x - 0.3, 0, target_z + lift_height)
        self.get_logger().info("Step 5: Object lifted (hover position).")
        time.sleep(3.0)
        



    def move_to_pose(self, x, y, z, roll=0.0, pitch=np.pi/2, yaw=0.0):
        if self.use_sim:
            # Simulated behavior
            self.get_logger().info(f"Simulated behavior: Moving in simulation to Pose ({x}, {y}, {z})")
            goal_msg = MoveArm.Goal()
            goal_msg.pose = [x, y, z, 0.0, 0.0, 0.0]  # Assuming no rotation for simplicity

            self.send_goal_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
            self.send_goal_future.add_done_callback(self.goal_response_callback)
        else:
            # Actual behavior
            self.get_logger().info(f"Real behavior: Moving hardware to Pose ({x}, {y}, {z})")

            # Convert roll, pitch, yaw to a rotation matrix
            r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            rotation_matrix = r.as_matrix()

            matrix = np.eye(4)
            matrix[:3, :3] = rotation_matrix  # Set rotation
            matrix[0, 3] = x
            matrix[1, 3] = y
            matrix[2, 3] = z

            self.locobot.arm.set_ee_pose_matrix(matrix, execute=True)

    def control_gripper(self, open_gripper):
        if self.use_sim:
            goal_msg = MoveGripper.Goal()
            if open_gripper:
                goal_msg.command = 'open'
                goal_msg.duration = 3.0
            else:
                goal_msg.command = 'closed'
                goal_msg.duration = 3.0
            self.send_gripper_future = self.__gripper_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
            self.send_gripper_future.add_done_callback(self.gripper_response_callback)
        else:
            if open_gripper:
                self.locobot.gripper.release()
            else:
                self.locobot.gripper.grasp()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal Rejected')
            return 
        
        self.get_logger().info('Goal Accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
    
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Received feedback: {feedback_msg.feedback.progress}")
        
    def gripper_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal Rejected')
            return 
        
        self.get_logger().info('Goal Accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def gripper_callback(self, msg: Bool):
        # Log the gripper state request
        self.get_logger().info(f"Received Gripper Command: {'Open' if msg.data else 'Close'}")
        
        # Control the gripper based on the received message
        self.control_gripper(msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = ArmWrapperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

