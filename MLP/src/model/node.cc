
#include "node.h"

namespace s21 {

Node::Node(std::size_t neurons) {
  BuildOfWeights(neurons, neurons);
  bias_ = Eigen::VectorXd::Random(neurons).array() * 0.1;
}

auto Node::BuildOfWeights(const std::size_t neurons, const std::size_t inputs)
    -> void {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);
  auto scaling_factor = std::sqrt(2.0 / (inputs + neurons));
  velocity_ = Eigen::MatrixXd::Zero(neurons, inputs);
  weights_ = Eigen::MatrixXd::NullaryExpr(
      neurons, inputs, [&]() { return dis(gen) * scaling_factor; });
}

auto Node::Sigmoid(double value) -> double { return (1 + std::exp(-value)); }

auto Node::Derivative() -> double { return value_ * (1 - value_); }

auto Node::Summator(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  return weights_ * inputs;
}

auto Node::Activate(Eigen::VectorXd inputs) -> double {
  value_ = (Summator(inputs) + bias_).sum();
  return Sigmoid(value_);
}

}  // namespace s21
