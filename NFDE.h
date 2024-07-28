#ifndef NFDE_H
#define NFDE_H

//Standard C++ librairies
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>

//ROS2 librairies
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class NFDE : public rclcpp::Node 
{
public:
  NFDE();

private:
  // Initialization routine:
  //    - Build the kernel matrix A
  //    - Initialize the empty vectors 
  void initialize();

  // Piece-wise linear version of the shifted tanh:
  //    - argument : input of the function 
  //    - satmin   : minimum saturation value
  //    - change   : x-coordinate of the point at which the slope changes from one to a higher value
  //    - slope    : slope of the third part of the function
  //    - satmax   : maximum saturation value 
  double S(double argument, double satmin, double change, double slope, double satmax);

  // Callback of the input topic, taking as input the processed LIDAR infos
  // (or any other inputs compatible with the size of the Neural field)
  // Apply the neural field equations and then call publish_solution
  void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  // Publish the neural field state on the output topic
  void publish_solution(const std::vector<double>& solution);
  
  // Publisher/Subscriber 
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr input_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr output_publisher_;

  // Class variables
  size_t count_;
  int N;
  double dt;
  Eigen::VectorXd previous_solution;
  Eigen::MatrixXd A; 
};

#endif // NFDE_H
