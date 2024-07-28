# Node estimating the position of the robot thanks to the cartographer node, by chaining the transforms

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from tf_transformations import quaternion_matrix, quaternion_from_matrix
import tf_transformations as tf
import numpy as np


class PosePublisher(Node):

    def __init__(self):
        super().__init__('pose_publisher')
        
        # TRANSFORMS
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TIMER 
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # PUBLISHER 
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/pose_topic',
            10
        )
        


    def timer_callback(self):
        try:
            # GETTING THE TRANSFORMS 
            self.get_logger().info("Receiving transforms...")
            trans1 = self.tf_buffer.lookup_transform('base_footprint', 'base_link', rclpy.time.Time())
            trans2 = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
            trans3 = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            self.get_logger().info("All transforms have been found")

            robot_pose = PoseStamped()
            robot_pose.header.frame_id = 'map'
            robot_pose.header.stamp = self.get_clock().now().to_msg()

            # TRANSFORMING THEM TO MATRICES
            trans1_matrix = self.transform_to_matrix(trans1.transform)
            trans2_matrix = self.transform_to_matrix(trans2.transform)
            trans3_matrix = self.transform_to_matrix(trans3.transform)

            # COMPUTING FINAL POSE
            map_pose_matrix = trans3_matrix @ trans2_matrix @ trans1_matrix
            robot_pose.pose = self.matrix_to_pose(map_pose_matrix)

            self.get_logger().info("Publishing final pose")
            self.pose_publisher.publish(robot_pose)
            
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().error('Transform lookup failed: %s' % str(e))



    def transform_to_matrix(self, transform):
        trans = [transform.translation.x, transform.translation.y, transform.translation.z]
        rot = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        matrix = quaternion_matrix(rot)
        matrix[:3, 3] = trans
        return matrix

    def matrix_to_pose(self, matrix):
        trans = matrix[:3, 3]
        rot = quaternion_from_matrix(matrix)
        pose = Pose()
        pose.position.x = trans[0]
        pose.position.y = trans[1]
        pose.position.z = trans[2]
        pose.orientation.x = rot[0]
        pose.orientation.y = rot[1]
        pose.orientation.z = rot[2]
        pose.orientation.w = rot[3]
        return pose

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
