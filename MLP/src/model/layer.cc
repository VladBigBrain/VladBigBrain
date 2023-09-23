#include "layer.h"

namespace s21 {

Layer::Layer(std::size_t neurons, std::size_t inputs) {
  for (auto i = 0; i < neurons; ++i) {
    neurons_.push_back(Neuron(0));
  }
  bias_ = Eigen::VectorXd::Random(neurons).array() * 0.1;
  BuildMatrixOfWeights(inputs);
  velocity_ = Eigen::MatrixXd::Zero(neurons_.size(), inputs);
}

auto Layer::FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd {
  Eigen::VectorXd output_vector = weights_ * inputs;
  return BuildNeurons(bias_ + output_vector);
}

auto Layer::BuildNeurons(const Eigen::VectorXd &out) -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(neurons_.size(), [&](Eigen::Index i) {
    return neurons_[i].Activate(out(i));
  });
}

auto Layer::BackPropagation(const Eigen::VectorXd &error, double learningRate,
                            Layer &layer) -> Eigen::VectorXd {
  Eigen::VectorXd gradient = error.array() * GetDerivativeVector().array();
  Eigen::MatrixXd deltaweights =
      learningRate * gradient * layer.GetOutputNeurons().transpose();
  Eigen::VectorXd newerror = weights_.transpose() * gradient;
  double gamma = 0.9;
  velocity_ = gamma * velocity_ + deltaweights;
  weights_ += velocity_;
  bias_ += learningRate * gradient;
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

const Eigen::MatrixXd &Layer::GetVelocity() const { return velocity_; }

void Layer::SetVelocity(const Eigen::MatrixXd &newVelocity) {
  velocity_ = newVelocity;
}

const Eigen::VectorXd &Layer::GetBias() const { return bias_; }

auto Layer::Size() const -> size_t { return neurons_.size(); }

auto Layer::BuildMatrixOfWeights(const std::size_t inputs) -> void {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);

  auto scaling_factor = std::sqrt(2.0 / (inputs + neurons_.size()));

  weights_ = Eigen::MatrixXd::NullaryExpr(
      neurons_.size(), inputs, [&]() { return dis(gen) * scaling_factor; });
}

auto operator<<(std::ostream &os, const Layer &layer) -> std::ostream & {
  os << layer.weights_.rows() << " " << layer.weights_.cols() << std::endl;
  os << layer.weights_ << std::endl;
  return os;
}

auto operator>>(std::ifstream &is, Layer &layer) -> std::ifstream & {
  int rows, cols;
  if (is >> rows >> cols) {
    layer.weights_ = Eigen::MatrixXd(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double weight;
        if (is >> weight) {
          layer.weights_(i, j) = weight;
        }
      }
    }
  }
  return is;
}

}  // namespace s21
