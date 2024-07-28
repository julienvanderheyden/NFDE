# Node processing the LIDAR infos and sending them through the "input_topic" topic

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from rclpy.qos import ReliabilityPolicy, QoSProfile
import numpy as np


def yaw_from_quaternion(qx, qy, qz, qw):
    norm = np.sqrt(qx**2 +  qy**2 +qz**2 + qw**2)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    
    yaw = np.arctan2(2*(qz*qw + qx*qy), 1-2*(qy**2 + qz**2))
    
    return yaw


class LaserScanSubscriber(Node):
    def __init__(self):
        super().__init__('laser_scan_subscriber')

        # Publishers/Subscribers 
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',  # Laser scan topic
            self.scan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))  # QoS profile
        
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/pose_topic', # Topic sending an estimation of  the pose of the robot, useful to compute the target excitation
            self.pose_callback,
            10)

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/input_topic',  # Input topic for the NFDE node 
            10)  # QoS profile

        # Create a timer to publish input data every 10ms
        self.timer = self.create_timer(0.01, self.publish_input_data)

        # Class Variables 
        self.N = 101 #should be odd such that the point zero is considered
        dev = 0.05 #to avoid un unstable equilibirum at pi
        self.angles = np.linspace(-np.pi + dev, np.pi -dev,self.N)  # Angles from -pi to pi
        self.angle_step = (2 * np.pi - 2*dev)/ self.N  # Angle increment in self.angles

        self.max_distance = 10.0
        self.distances = np.zeros(self.N)  # Initialize distances to 0
        self.distances_count = np.zeros(self.N, dtype=int)  # Initialize count of received distances to 0
        
        self.forward_excitation = np.exp(-self.angles**2 / (2*0.5**2))
        self.forward_excitation /= np.max(self.forward_excitation)
        
        #Keep track of the robot position and orientation to make the target excitation
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Target position 
        self.xtarget = 4.0 #to define before running the program
        self.ytarget = 1.0 #to define before running the program
        
        
    def pose_callback(self,msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        
        self.yaw = yaw_from_quaternion(qx, qy, qz, qw)


    def scan_callback(self, msg):
        # Extract distances from laser scan data
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        # Reset distances array and count array
        self.distances.fill(0.0)
        self.distances_count.fill(0.0)
        
        # Process laser scan data
        for i, distance in enumerate(msg.ranges):

            # Compute angle for current point
            angle = angle_min + i * angle_increment

            if 0 <= angle <= np.pi:
                angle = - angle
            else :
                angle = 2*np.pi - angle

            # Find closest angle index
            closest_angle_index = np.argmin(np.abs(angle-self.angles))

            if not np.isfinite(distance):  # Only process finite distances
                distance = self.max_distance
                
            # Add distance to corresponding angle
            self.distances[closest_angle_index] += distance
            self.distances_count[closest_angle_index] += 1

        # Compute final distances 
        for i, count in enumerate(self.distances_count):
            if count > 0:
                self.distances[i] /= count
                self.distances[i] = round(self.distances[i],3)

    def publish_input_data(self):

        #process distances thanks to the chosen process_input function
        self.distances = self.process_input(self.distances)

        # Publish distances as Float64MultiArray
        array_msg = Float64MultiArray()
        array_msg.data = self.distances.tolist()
        self.publisher.publish(array_msg)
        
        # For debugging
        # self.get_logger().info('Publishing vector "%s"' % array_msg.data)

    # Change this function if you want to change the link between distances and Neural Field Input (tanh, exp, ...)
    def process_input(self, distances):

        threshold = 0.85
        inhibitory_strength = 1.0
        excitatory_strength = 1.2
        
        target_excitation = np.zeros(self.N)
        
        target_angle = self.compute_target_angle()
        
        # Excitatory Part 
        for i in range(self.N):
            target_excitation[i] = excitatory_strength*0.5*(np.cos(self.angles[i] - target_angle)+1)
            
        # Inhibitory Part 
        for i in range(self.N):
            if distances[i] < threshold:
                distances[i] = -inhibitory_strength
        
            else:
                distances[i] = target_excitation[i]
                
        return distances
    
    def compute_target_angle(self):
        target_vector = np.array([self.y - self.ytarget, self.xtarget - self.x])
        target_angle = np.arctan2(target_vector[1], target_vector[0])
        
        yaw = self.yaw
        
        #ensure yaw in range [-pi, pi]
        yaw = yaw % (2*np.pi)
        if yaw > np.pi : 
            yaw -= 2*np.pi
            
        angle_difference = np.pi/2 + yaw - target_angle
        
        #ensure difference in range [-pi,pi]
        if angle_difference > np.pi:
            angle_difference -= 2*np.pi
        elif angle_difference <  -np.pi:
            angle_difference += 2*np.pi
            
        return angle_difference
    

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
