#include "layer.h"

#include "console.h"
namespace s21 {
Layer::Layer(std::size_t neurons, std::size_t inputs) {
  for (auto i = 0; i < neurons; ++i) {
    neurons_.push_back(Neuron(0, 0));
  }
  BuildMatrixOfWeights(inputs);
}

auto Layer::FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  Eigen::VectorXd output = weights_ * inputs;
  BuildNeurons(output);
  return output;
}

auto Layer::BackPropagation(const Eigen::VectorXd& error, double learningRate)
    -> Eigen::VectorXd {
Eigen::VectorXd gradient = error.array() * GetDerivativeVector().array();
  Eigen::MatrixXd deltaWeights = learningRate * gradient * weights_.transpose();
  weights_ -= deltaWeights;
  Eigen::VectorXd prevError = weights_.transpose() * gradient;
  return prevError;
}

auto Layer::GetDerivativeVector() -> Eigen::VectorXd {
  Eigen::VectorXd out(neurons_.size());
  for (auto i = 0; i < neurons_.size(); ++i) {
    out(i) = neurons_[i].Derivative();
  }
  return out;
}

auto Layer::GetNeurons() const -> std::vector<Neuron> { return neurons_; }
auto Layer::Size() const -> size_t { return neurons_.size(); }

auto Layer::BuildMatrixOfWeights(const std::size_t inputs) -> void {
  weights_ = Eigen::MatrixXd::Random(neurons_.size(), inputs);
}

auto Layer::BuildNeurons(const Eigen::VectorXd& out) -> void {
  for (auto i = 0; i < neurons_.size(); ++i) {
    neurons_[i].Activate(out(i));
  }
}

auto operator<<(std::ostream& os, const Layer& layer) -> std::ostream& {
  for (const auto& neuron : layer.neurons_) {
    os << neuron;
  }
  os << layer.weights_;
  std::cout << std::endl;
  return os;
}

}  // namespace s21
