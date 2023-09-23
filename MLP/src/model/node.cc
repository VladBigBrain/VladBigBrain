
#include "node.h"

namespace s21 {

Node::Node(std::size_t neurons) {
  BuildOfWeights(neurons);
  bias_ = Eigen::VectorXd::Random(neurons).array() * 0.1;
}

auto Node::BuildOfWeights(const std::size_t neurons) -> void {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);
  auto scaling_factor = std::sqrt(2.0 / (neurons + neurons));
  velocity_ = Eigen::VectorXd::Zero(neurons);
  weights_ = Eigen::VectorXd::NullaryExpr(
      neurons, [&]() { return dis(gen) * scaling_factor; });
}

auto Node::Sigmoid(double value) -> double { return (1 + std::exp(-value)); }

auto Node::Derivative() -> double { return value_ * (1 - value_); }

auto Node::Activate(Eigen::VectorXd inputs) -> double {
  value_ = Sigmoid(weights_.dot(inputs) + bias_.sum());
  return value_;
}

}  // namespace s21
