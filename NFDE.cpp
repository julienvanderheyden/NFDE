#include "NFDE.h"

using std::placeholders::_1;
using namespace std::chrono_literals;

NFDE::NFDE() 
    : Node("NFDE"), count_(0), N(101), dt(1.0), previous_solution(N), A(N, N)
    // N : size of the neural field 
    // dt : time step  
{
  // Change this two lines you use other input/output topics
  output_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("output_topic", 10);
  input_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "input_topic", 10, std::bind(&NFDE::topic_callback, this, _1));

  initialize();
}

void NFDE::topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // Start timing the callback execution
  rclcpp::Time start_time = this->now();

  // Map input message data to Eigen vector
  Eigen::VectorXd inputs = Eigen::Map<Eigen::VectorXd>(msg->data.data(), msg->data.size());
  Eigen::VectorXd argument = Eigen::Map<Eigen::VectorXd>(previous_solution.data(), previous_solution.size());

  // Perform the computation
  float kbar = 1.2;
  argument = kbar * (A * argument + inputs);
  auto wrapper = [&](double x) { return S(x, -1.0, 1.0, 2.0, 3.0); }; // Change this line if you want to change the tanh parameters
  argument = argument.unaryExpr(wrapper);

  // Calculate z_dot and update solution
  Eigen::VectorXd z_dot = -previous_solution + argument;
  Eigen::VectorXd solution = previous_solution + dt * z_dot; //Euler step
  previous_solution = solution;

  // Convert solution to vector for publishing
  std::vector<double> solution_vector(solution.data(), solution.data() + solution.size());

  // End timing the callback execution
  rclcpp::Time end_time = this->now();
  rclcpp::Duration duration = end_time - start_time;
  RCLCPP_INFO(this->get_logger(), "Operation took %f seconds", duration.seconds());

  // Publish the solution
  publish_solution(solution_vector);
}

void NFDE::publish_solution(const std::vector<double>& solution) 
{
  auto message = std_msgs::msg::Float64MultiArray();
  message.data = solution;

  // Print output for debugging 
  RCLCPP_INFO(this->get_logger(), "Publishing Neural Field State:");
  for (size_t i = 0; i < solution.size(); ++i) {
    RCLCPP_INFO(this->get_logger(), "NFDE[%zu]: %f", i, solution[i]);
  }

  // Publish output message 
  output_publisher_->publish(message);
}

void NFDE::initialize() {
  // Initialize previous_solution vector and matrix A with specific values

  // c vector is the base vector of the A circulant matrix. Must be re-computed for other kernels or kernel size. 
  std::vector<double> c = {0.20653625302992937, 0.14818107237361877, 0.0744316148426165, 0.03851575839563159, 
                           0.019328549351016043, 0.01001793528832492, 0.0036590577590086128, 0.0004178615641413476, 
                           -0.0024616045361125004, -0.0038071009028551174, -0.0054190596040954294, -0.0060122542042517066, 
                           -0.00705841387418237, -0.0073020829886658355, -0.008056555978476494, -0.00812000641637229, 
                           -0.008706647110583032, -0.008670825536023693, -0.009151800129914667, -0.009059412059768046, 
                           -0.009468453018845337, -0.009343872930060091, -0.00970038850343738, -0.00955846532588002, 
                           -0.00987412485897108, -0.009724420555792847, -0.010006472021480141, -0.009855446355304226, 
                           -0.010108495625828936, -0.009960697514781278, -0.010187713832191466, -0.01004646224672262, 
                           -0.010249372143515305, -0.010117160974802992, -0.010297212650377984, -0.010175959658200265, 
                           -0.010333953931041508, -0.010225158220882234, -0.0103615990434294, -0.01026644306667591, 
                           -0.010381637915627701, -0.010301054802168867, -0.010395182847908625, -0.010329901463427786, 
                           -0.010403060380714091, -0.010353635670508959, -0.010405873819845311, -0.010372707135301695, 
                           -0.010404045336257023, -0.010387397680124822, -0.010397843217915207, -0.010397843217915207, 
                           -0.010387397680124822, -0.010404045336257023, -0.010372707135301695, -0.010405873819845311, 
                           -0.010353635670508956, -0.010403060380714094, -0.010329901463427786, -0.010395182847908628, 
                           -0.010301054802168867, -0.0103816379156277, -0.01026644306667591, -0.0103615990434294, 
                           -0.010225158220882232, -0.010333953931041508, -0.010175959658200267, -0.010297212650377984, 
                           -0.010117160974802992, -0.010249372143515305, -0.01004646224672262, -0.010187713832191464, 
                           -0.009960697514781278, -0.010108495625828941, -0.009855446355304226, -0.010006472021480143, 
                           -0.009724420555792848, -0.009874124858971078, -0.009558465325880018, -0.009700388503437381, 
                           -0.009343872930060093, -0.009468453018845333, -0.009059412059768044, -0.009151800129914664, 
                           -0.00867082553602369, -0.00870664711058303, -0.00812000641637229, -0.008056555978476494, 
                           -0.007302082988665833, -0.007058413874182366, -0.006012254204251701, -0.005419059604095432, 
                           -0.003807100902855119, -0.0024616045361125012, 0.00041786156414134486, 0.0036590577590086115, 
                           0.01001793528832492, 0.019328549351016043, 0.03851575839563159, 0.0744316148426165, 
                           0.14818107237361877};

  // Build A matrix based on the c vector
  for (int i = 0; i < N; ++i) {
    previous_solution(i) = 0.0;
    for (int j = 0; j < N; ++j) {
      A(i, j) = c[(N + j - i) % N];
    }
  }

  // Print A matrix for debugging
  RCLCPP_INFO(this->get_logger(), "Kernel matrix initialized :");
  std::ostringstream oss;
  oss << A;
  RCLCPP_INFO(this->get_logger(), oss.str().c_str());
}
