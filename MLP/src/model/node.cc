
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
  os << node.weights_.size() << ' ';
  os << node.weights_.transpose() << std::endl; // Используем возможности Eigen
  os << "---" << std::endl;                     // Разделитель
  return os;
}

auto operator>>(std::ifstream &is, Node &node) -> std::ifstream & {
  std::size_t size;
  is >> size;

  Eigen::VectorXd temp(size);
  for (std::size_t i = 0; i < size; ++i) {
    is >> temp[i];
  }

  std::string delimiter;
  std::getline(is, delimiter); // Съедаем оставшийся перевод строки
  std::getline(is, delimiter); // Читаем разделитель

  if (delimiter != "---") {
    // Handle error
  }

  node.SetWeights(temp);
  return is;
}

} // namespace s21
