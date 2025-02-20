#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data
import numpy as np


class LocobotExample(Node):
    """Low-level control node: Robot moves based solely on target poses received from command node."""
    def __init__(self):
        super().__init__('move_A_to_B_py')

        # Target pose will now only be updated from command node
        self.target_pose = None

        # Publishers
        self.mobile_base_vel_publisher = self.create_publisher(Twist, "/locobot/diffdrive_controller/cmd_vel_unstamped", 1)
        self.point_P_control_point_visual = self.create_publisher(Marker, "/locobot/mobile_base/control_point_P", 1)
        self.target_pose_visual = self.create_publisher(Marker, "/locobot/mobile_base/target_pose_visual", 1)

        # Subscribers
        self.odom_subscription = self.create_subscription(
            Odometry,
            "/locobot/sim_ground_truth_pose",
            self.odom_mobile_base_callback,
            qos_profile_sensor_data
        )

        self.target_pose_subscription = self.create_subscription(
            Pose,
            '/target_pose',
            self.target_pose_callback,
            10
        )

        self.L = 0.1
        self.goal_reached_error = 0.05
        self.integrated_error_list = []
        self.length_of_integrated_error_list = 20

    def target_pose_callback(self, msg):
        """Update target pose from command node."""
        self.get_logger().info(f"Received new target pose: x={msg.position.x}, y={msg.position.y}")
        self.target_pose = msg

    def pub_point_P_marker(self):
        marker = Marker()
        marker.header.frame_id = "locobot/base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.pose.position.x = self.L
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.point_P_control_point_visual.publish(marker)

    def pub_target_point_marker(self):
        if self.target_pose:
            marker = Marker()
            marker.header.frame_id = "locobot/odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 0
            marker.type = Marker.ARROW
            marker.scale.x = 0.3
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.pose = self.target_pose
            marker.color.a = 1.0
            marker.color.g = 1.0
            self.target_pose_visual.publish(marker)

    def odom_mobile_base_callback(self, data):
        if not self.target_pose:
            self.get_logger().info("Waiting for target pose from command node...")
            return

        x_data, y_data = data.pose.pose.position.x, data.pose.pose.position.y
        qw, qx, qy, qz = data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z

        R11 = qw**2 + qx**2 - qy**2 - qz**2
        R21 = 2 * (qx * qz - qw * qy)

        point_P = Point()
        point_P.x = x_data + self.L * R11
        point_P.y = y_data + self.L * R21

        self.pub_point_P_marker()
        self.pub_target_point_marker()

        err_x = self.target_pose.position.x - point_P.x
        err_y = self.target_pose.position.y - point_P.y
        error_vect = np.matrix([[err_x], [err_y]])

        if np.linalg.norm(error_vect) < self.goal_reached_error:
            self.get_logger().info("Target reached!")
            self.publish_zero_velocity()
            return

        Kp_mat = 1.2 * np.eye(2)
        current_angle = np.arctan2(R21, R11)
        non_holonomic_mat = np.matrix([[np.cos(current_angle), -self.L * np.sin(current_angle)],
                                       [np.sin(current_angle),  self.L * np.cos(current_angle)]])

        point_p_error_signal = Kp_mat @ error_vect
        control_input = np.linalg.inv(non_holonomic_mat) @ point_p_error_signal

        control_msg = Twist()
        control_msg.linear.x = float(control_input.item(0))
        control_msg.angular.z = float(control_input.item(1))
        self.mobile_base_vel_publisher.publish(control_msg)

    def publish_zero_velocity(self):
        """Stop the robot by publishing zero velocity."""
        control_msg = Twist()
        control_msg.linear.x = 0.0
        control_msg.angular.z = 0.0
        self.mobile_base_vel_publisher.publish(control_msg)
        self.get_logger().info("Robot stopped at target.")


def main(args=None):
    rclpy.init(args=args)
    node = LocobotExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


