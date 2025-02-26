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

class StateMachine:
    """
    A simple four-step state machine to pick an object:
      1) move above the target (+0.5m z-offset),
      2) open gripper,
      3) move to target,
      4) close gripper.
    """
    def __init__(self, arm_node):
        """
        :param arm_node: An instance of ArmWrapperNode (provides methods to move the arm).
        """
        self.arm_node = arm_node
        self.current_state = "IDLE"
        self.target_pose = None           # 3-element np.array [x, y, z]
        self.target_orientation = None    # geometry_msgs.msg.Quaternion

        # How high above the target (in meters) we want to hover before descending.
        self.z_offset = 0.3

    def set_target(self, position, orientation):
        """
        Sets the pick target and transitions into the picking sequence.
        :param position: geometry_msgs.msg.Point (x, y, z)
        :param orientation: geometry_msgs.msg.Quaternion (x, y, z, w)
        """
        self.target_pose = np.array([position.x, position.y, position.z])
        self.target_orientation = orientation
        self.current_state = "MOVE_ABOVE_TARGET"
        self.run()

    def run(self):
        """Run the next step in the state machine based on `current_state`."""
        if self.current_state == "MOVE_ABOVE_TARGET":
            above_pose = self._calculate_above_pose()
            self.arm_node.get_logger().info("StateMachine: Moving above target...")
            self.arm_node.move_to_pose(above_pose, self.target_orientation, self.on_movement_complete)

        elif self.current_state == "OPEN_GRIPPER":
            self.arm_node.get_logger().info("StateMachine: Opening gripper...")
            self.arm_node.open_gripper(self.on_movement_complete)

        elif self.current_state == "MOVE_TO_TARGET":
            self.arm_node.get_logger().info("StateMachine: Moving to target...")
            self.arm_node.move_to_pose(self.target_pose, self.target_orientation, self.on_movement_complete)

        elif self.current_state == "CLOSE_GRIPPER":
            self.arm_node.get_logger().info("StateMachine: Closing gripper...")
            self.arm_node.close_gripper(self.on_movement_complete)

        elif self.current_state == "IDLE":
            self.arm_node.get_logger().info("StateMachine: Task completed!")

    def _calculate_above_pose(self):
        """
        Adds a +0.5m Z offset to the target pose to position the gripper 'above' the item.
        """
        if self.target_pose is None:
            raise ValueError("StateMachine: No target pose set before _calculate_above_pose!")
        pose_above = self.target_pose.copy()
        pose_above[2] += self.z_offset
        return pose_above

    def on_movement_complete(self):
        """
        Callback from ArmWrapperNode once a move or gripper action is done.  
        Advances the state machine to the next step.
        """

        # Add a delay between actions
        delay_seconds = 1.0  # Delay for 2 seconds (adjust as needed)
        self.arm_node.get_logger().info(f"Waiting for {delay_seconds} seconds before next action...")
        time.sleep(delay_seconds)
        
        if self.current_state == "MOVE_ABOVE_TARGET":
            self.current_state = "OPEN_GRIPPER"
        elif self.current_state == "OPEN_GRIPPER":
            self.current_state = "MOVE_TO_TARGET"
        elif self.current_state == "MOVE_TO_TARGET":
            self.current_state = "CLOSE_GRIPPER"
        elif self.current_state == "CLOSE_GRIPPER":
            self.current_state = "IDLE"
        # Move on to the next step
        self.run()


class ArmWrapperNode(Node):
    """
    Node that 'wraps' arm control. 
    It can command either the hardware (via InterbotixLocobotXS) or a simulation (via ROS actions).
    """
    def __init__(self):
        super().__init__('arm_wrapper_node')
        
        # Declare and read the 'use_sim' parameter
        self.declare_parameter('use_sim', False)
        self.use_sim = self.get_parameter('use_sim').get_parameter_value().bool_value
        self.get_logger().info(f"use_sim parameter set to: {self.use_sim}")

        # Create subscriptions
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

        # Create action clients for simulation
        self._action_client = ActionClient(self, MoveArm, 'movearm')
        self._gripper_client = ActionClient(self, MoveGripper, 'movegripper')

        # If not in sim, initialize real locobot hardware interface
        if not self.use_sim:
            self.locobot = InterbotixLocobotXS(
                robot_model='locobot_wx250s',
                robot_name='locobot',
                arm_model='mobile_wx250s'
            )

        # Create the StateMachine and pass this node as the interface
        self.state_machine = StateMachine(self)

    # ------------------------
    # Subscriber Callbacks
    # ------------------------
    def pose_callback(self, msg: PoseStamped):
        """
        When we receive a PoseStamped, we interpret it as a new pick target.
        We hand it over to the state machine to begin the pick sequence.
        """
        self.get_logger().info(f"Received Pose: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")
        self.state_machine.set_target(msg.pose.position, msg.pose.orientation)

    def gripper_callback(self, msg: Bool):
        """
        If a Bool message arrives:
          True  => open the gripper
          False => close the gripper
        This direct callback can override the state machine behavior 
        (e.g., for manual control), or can be left unused.
        """
        self.get_logger().info(f"Received Gripper Command: {msg.data}")
        if self.use_sim:
            goal_msg = MoveGripper.Goal()
            if msg.data:
                goal_msg.command = 'open'
            else:
                goal_msg.command = 'closed'
            goal_msg.duration = 3.0
            
            send_future = self._gripper_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
            send_future.add_done_callback(self.gripper_response_callback)
        else:
            if msg.data:
                self.locobot.gripper.release()
            else:
                self.locobot.gripper.grasp()

    # ------------------------
    # StateMachine Helpers
    # (Called by the StateMachine)
    # ------------------------
    def move_to_pose(self, position, orientation, done_cb):
        """
        In simulation: send a MoveArm action goal.
        In hardware: directly command the locobot arm.
        :param position: 3-element np.array
        :param orientation: geometry_msgs.msg.Quaternion
        :param done_cb: callback to invoke when movement is complete
        """
        if self.use_sim:
            # Convert quaternion to Euler
            r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            eul = r.as_euler('xyz', degrees=True)

            # Construct action goal
            goal_msg = MoveArm.Goal()
            goal_msg.pose = [position[0], position[1], position[2], eul[0], eul[1], eul[2]]

            # Send action goal
            send_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
            # pass `done_cb` through our standard result callback
            send_future.add_done_callback(lambda fut: self.goal_response_callback(fut, done_cb))

        else:
            # For real hardware, set the EE pose matrix directly
            r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            rot_mat = r.as_matrix()
            matrix = np.eye(4)
            matrix[0:3, 0:3] = rot_mat
            matrix[0, 3] = position[0]
            matrix[1, 3] = position[1]
            matrix[2, 3] = position[2]

            self.locobot.arm.set_ee_pose_matrix(matrix, execute=True)
            # Immediately invoke done_cb (no action server)
            done_cb()

    def open_gripper(self, done_cb):
        """
        Opens the gripper (hardware or simulation).
        :param done_cb: callback invoked when open action is complete
        """
        if self.use_sim:
            goal_msg = MoveGripper.Goal()
            goal_msg.command = 'open'
            goal_msg.duration = 3.0
            future = self._gripper_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
            future.add_done_callback(lambda fut: self.gripper_response_callback(fut, done_cb))
        else:
            self.locobot.gripper.release()
            done_cb()

    def close_gripper(self, done_cb):
        """
        Closes the gripper (hardware or simulation).
        :param done_cb: callback invoked when close action is complete
        """
        if self.use_sim:
            goal_msg = MoveGripper.Goal()
            goal_msg.command = 'closed'
            goal_msg.duration = 3.0
            future = self._gripper_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
            future.add_done_callback(lambda fut: self.gripper_response_callback(fut, done_cb))
        else:
            self.locobot.gripper.grasp()
            done_cb()

    # ------------------------
    # Action Callbacks
    # ------------------------
    def goal_response_callback(self, future, done_cb):
        """
        Called when the action server responds with acceptance/rejection of the goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal Rejected')
            # (Optionally call done_cb or handle error)
            return

        self.get_logger().info('Goal Accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self.get_result_callback(fut, done_cb))

    def get_result_callback(self, future, done_cb):
        """
        Called when the action finishes and provides a result.
        """
        _ = future.result().result
        # Movement or gripper command is complete
        done_cb()

    def gripper_response_callback(self, future, done_cb=None):
        """
        Similar to goal_response_callback, but for the gripper action.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper Goal Rejected')
            return

        self.get_logger().info('Gripper Goal Accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self.get_result_callback(fut, done_cb) 
                                        if done_cb else None)

    def feedback_callback(self, feedback_msg):
        """
        Periodic feedback from action server (both MoveArm and MoveGripper).
        """
        self.get_logger().info(f"Feedback: {feedback_msg.feedback.progress}")


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