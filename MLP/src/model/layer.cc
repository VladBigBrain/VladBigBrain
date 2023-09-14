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

// Function Layer::BackPropagation(double error, double learningRate):
//     # Шаг 1: Вычислить градиент ошибки для текущего слоя
//     gradient = DerivativeOfActivation(this->neurons_value) * error

//     # Шаг 2: Обратно распространить ошибку на предыдущий слой
//     If this is not InputLayer:
//         previous_layer_error = weights_matrix.Transpose() * gradient

//     # Шаг 3: Обновление весов текущего слоя
//     weights_matrix -= learningRate * gradient

//     # Возвращаем ошибку для предыдущего слоя, чтобы продолжить цикл
//     return previous_layer_error

auto Layer::BackPropagation(const Eigen::VectorXd& inputs, double error,
                            double learningRate) -> Eigen::VectorXd {
  Eigen::VectorXd gradient(neurons_.size());
  
  for (size_t i = 0; i < neurons_.size(); ++i) {
    gradient(i) = neurons_[i].Derivative() * error;
  }
  
  Eigen::MatrixXd deltaWeights = learningRate * gradient * inputs.transpose();
  weights_ -= deltaWeights;
  return inputs;
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
