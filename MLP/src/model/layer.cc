#include "layer.h"

#include "console.h"
namespace s21 {
Layer::Layer(std::size_t neurons, std::size_t inputs) {
  for (auto i = 0; i < neurons; ++i) {
    neurons_.push_back(Neuron(0, 0));
  }
  BuildMatrixOfWeights(inputs);
}

auto Layer::FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd {
  Eigen::VectorXd output = weights_ * inputs;
  BuildNeurons(output);
  return output;
}

auto Layer::BackPropagation(const Eigen::VectorXd &error, double learningRate,
                            Layer &layer) -> Eigen::VectorXd {
  // local gradient for each neuron
  Eigen::VectorXd gradient =
      error.array() * GetDerivativeVector().array(); // 26

  // calc deltaweights
  Eigen::MatrixXd deltaweights =
      gradient * layer.GetOutputNeurons().transpose() * learningRate; // x / 26

  // calc error output
  Eigen::VectorXd newerror = weights_.transpose() * gradient;

  // update weights
  weights_ += deltaweights;

  // return new error
  return newerror;
}

auto Layer::GetOutputNeurons() -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(neurons_.size(), [this](Eigen::Index i) {
    return neurons_[i].GetValue();
  });
}

void Layer::SetWeights(const Eigen::MatrixXd &weights) {
    weights_ = weights;
}

auto Layer::GetDerivativeVector() -> Eigen::VectorXd {
    return Eigen::VectorXd::NullaryExpr(neurons_.size(), [this](Eigen::Index i) {
    return neurons_[i].Derivative();
  });
}

auto Layer::GetNeurons() const -> std::vector<Neuron> { return neurons_; }
auto Layer::Size() const -> size_t { return neurons_.size(); }

auto Layer::BuildGradientMatrix(const Eigen::VectorXd &error)
    -> Eigen::MatrixXd {
  return weights_ * error;
}

auto Layer::BuildMatrixOfWeights(const std::size_t inputs) -> void {
  weights_ = Eigen::MatrixXd::Random(neurons_.size(), inputs);
}

auto Layer::BuildNeurons(const Eigen::VectorXd &out) -> void {
  for (auto i = 0; i < out.size(); ++i) {
    neurons_[i].Activate(out(i));
  }
}

auto operator<<(std::ostream &os, const Layer &layer) -> std::ostream & {
  for (const auto &neuron : layer.neurons_) {
    os << neuron;
  }
  os << layer.weights_;
  std::cout << std::endl;
  return os;
}

} // namespace s21
