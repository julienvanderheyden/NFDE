# Node taking as input the neural field state and controlling the robot motion 

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import ReliabilityPolicy, QoSProfile
import numpy as np


class OutputNode(Node):
    def __init__(self):
        super().__init__('output_node')
        
        # Publishers/Subscribers
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/output_topic',  # Neural field state topic
            self.output_callback,
            10)  # QoS profile
        
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',  # Command velocity topic
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))  # QoS profile

        self.N = 101  # Number of neural field elements
        
        self.omega = 0.0
        self.barv = 0.15
        self.dev = 0.05

        initial_twist = Twist()
        initial_twist.linear.x = self.barv
        initial_twist.angular.z = 0.0
        
        # Publish initial twist
        for i in range(50):
            self.publisher.publish(initial_twist)
            
        self.get_logger().info("Initial Twist published")
        


    def output_callback(self, msg):
        # Get the state of the neural field
        neural_field = np.array(msg.data)

        # Find the index of the maximum value in the neural field
        max_list = []
        max_value = max(neural_field)
        
        #Find all points where the neural field is maximal
        for i in range(self.N):
            if neural_field[i] == max_value:
                max_list.append(i)
        
        #Take the mean of them
        max_index = int(np.mean(max_list))

        # Calculate the angle corresponding to the max index
        angles = np.linspace(-np.pi + self.dev, np.pi - self.dev, self.N)
        max_angle = angles[max_index]
        
        # Add a dead zone around the angle 0° to avoid oscillations 
        if np.abs(max_angle) < 0.15 and neural_field[int(self.N/2)] == max_value:
            max_angle = 0.0

        # Create a Twist message to steer the robot
        twist = Twist()
        
        # Compute velocities
        k = 5.0
        equation_input = -k*np.sin(max_angle) 
        max_angular_velocity = 1.2
        omega_dot = - self.omega + max_angular_velocity*np.tanh(equation_input)
        dt = 0.1

        self.omega = self.omega + dt*omega_dot
        self.omega = round(self.omega, 4)
        
        # Add a dead zone to avoid oscillations 
        if np.abs(self.omega) < 0.05:
            self.omega = 0.0
            
        twist.angular.z = self.omega
        twist.linear.x = self.barv

        # Publish the Twist message
        self.publisher.publish(twist)
        self.get_logger().info(f"Steering {'left' if twist.angular.z > 0 else 'right'} to angle {round(max_angle,1)} with angular velocity {self.omega} ")

def main(args=None):
    rclpy.init(args=args)
    node = OutputNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
