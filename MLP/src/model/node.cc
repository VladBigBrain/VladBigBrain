
#include "node.h"

namespace s21 {

Node::Node(std::size_t neurons) {
  BuildOfWeights(neurons);
  bias_ = Eigen::VectorXd::Random(1).array() * 0.1;
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

auto Node::UpdateWeights(double gradient, double learningRate,
                         const Eigen::VectorXd &output) -> void {
  double gamma = 0.9;
  auto deltaweights = learningRate * gradient * output;
  velocity_ = gamma * velocity_ + deltaweights;
  weights_ += velocity_;
  bias_ = bias_.array() + learningRate * gradient;
}

double Node::GetValue() const { return value_; }

auto Node::GetWeights() const -> const Eigen::VectorXd & { return weights_; }

auto Node::Sigmoid(double value) -> double { return (1 + std::exp(-value)); }

auto Node::Derivative() -> double { return value_ * (1 - value_); }

auto Node::SetWeights(Eigen::VectorXd weights) -> void { weights_ = weights; }

auto Node::Activate(Eigen::VectorXd inputs) -> double {
  value_ = Sigmoid(weights_.dot(inputs) + bias_.sum());
  return value_;
}

auto operator<<(std::ostream &os, const Node &node) -> std::ostream & {
  for (int i = 0; i < node.weights_.size(); ++i) {
    os << node.weights_[i];
    if (i < node.weights_.size() - 1) {
      os << ", ";
    }
  }
  os << std::endl;
  return os;
}

auto operator>>(std::ifstream &is, Node &node) -> std::ifstream & {
  std::string line;
  if (std::getline(is, line)) {
    std::stringstream ss(line);
    std::vector<double> values;
    std::string item;
    while (std::getline(ss, item, ',')) { // Разделитель - запятая
      values.push_back(std::stod(item)); // Преобразование в double
    }
    node.SetWeights(Eigen::VectorXd::Map(values.data(), values.size()));
  }
  return is;
}

} // namespace s21
