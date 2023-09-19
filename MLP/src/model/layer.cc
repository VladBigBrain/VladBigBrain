#include "layer.h"
#include "console.h"
#include <cmath>
#include <random>
namespace s21 {
Layer::Layer(std::size_t neurons, std::size_t inputs) {
  for (auto i = 0; i < neurons; ++i) {
    neurons_.push_back(Neuron(0));
  }
  bias_ = Eigen::VectorXd::Random(neurons).array() * 0.1;
  BuildMatrixOfWeights(inputs);
}

auto Layer::FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd {
  Eigen::VectorXd output_vector = weights_ * inputs;
  return BuildNeurons(output_vector + bias_);
}

auto Layer::BuildNeurons(const Eigen::VectorXd &out) -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(neurons_.size(), [&](Eigen::Index i) {
    return neurons_[i].Activate(out(i));
  });
}

auto Layer::BackPropagation(const Eigen::VectorXd &error, double learningRate,
                            Layer &layer) -> Eigen::VectorXd {
  // local gradient for each neuron
  Eigen::VectorXd gradient =
      error.array() * GetDerivativeVector().array(); // 26
//  double momentum = 0.05;
  // calc deltaweights
  Eigen::MatrixXd deltaweights =
      gradient * layer.GetOutputNeurons().transpose() * learningRate ;
//          +
//      momentum * previous_deltaweights;
//  previous_deltaweights = deltaweights;
  // calc error output
  Eigen::VectorXd newerror = weights_.transpose() * gradient;

  // calc new bias
  bias_ += learningRate * gradient;
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

void Layer::SetWeights(const Eigen::MatrixXd &weights) { weights_ = weights; }

auto Layer::GetDerivativeVector() -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(neurons_.size(), [this](Eigen::Index i) {
    return neurons_[i].Derivative();
  });
}

auto Layer::GetNeurons() const -> std::vector<Neuron> { return neurons_; }

const Eigen::MatrixXd &Layer::getPrevious_deltaweights() const {
  return previous_deltaweights;
}

void Layer::setPrevious_deltaweights(
    const Eigen::MatrixXd &newPrevious_deltaweights) {
  previous_deltaweights = newPrevious_deltaweights;
}

const Eigen::VectorXd &Layer::bias() const { return bias_; }

auto Layer::Size() const -> size_t { return neurons_.size(); }

auto Layer::BuildGradientMatrix(const Eigen::VectorXd &error)
    -> Eigen::MatrixXd {
  return weights_ * error;
}

auto Layer::BuildMatrixOfWeights(const std::size_t inputs) -> void {
  //  double variance = sqrt(2.0 / (neurons_.size() + inputs));
  //  std::cerr << variance;
  weights_ = Eigen::MatrixXd::Random(neurons_.size(), inputs) * 0.5;
  previous_deltaweights = Eigen::MatrixXd::Zero(neurons_.size(), inputs);
  //  weights_ = Eigen::MatrixXd::Random(neurons_.size(), inputs) * variance;
  //  std::cerr << weights_;
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
