#ifndef MLP_MODEL_LAYER_H_
#define MLP_MODEL_LAYER_H_

#include <iostream>

#include "algorithm.h"
#include "neuron.h"
namespace s21 {

class Layer {
 public:
  Layer(std::size_t neurons) {
    for (auto i = 0; i < neurons; ++i) {
      neurons_.push_back(Neuron(0, neurons, 0));
    }
  }
  Layer(const Layer&) = default;
  Layer(Layer&&) = default;
  auto operator=(const Layer&) -> Layer& = default;
  auto operator=(Layer&&) -> Layer& = default;
  ~Layer() = default;
  auto FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd& gradients, double learningRate)
      -> void;
  [[nodiscard]] auto GetNeurons() const -> std::vector<Neuron> {
    return neurons_;
  }
  auto Size() const -> size_t { return neurons_.size(); }
  auto BuildMatrixOfWeights() -> Eigen::MatrixXd {
    Eigen::MatrixXd weights(neurons_.size(), neurons_[0].GetWeights().size());
    for (auto i = 0; i < neurons_.size(); ++i) {
      weights.row(i) = neurons_[i].GetWeights();
    }
    return weights;
  }

  auto BuildNeurons(const Eigen::VectorXd& out) -> void {
    for (auto i = 0; i < neurons_.size(); ++i) {
      neurons_[i](out(i));
    }
  }

  friend auto operator<<(std::ostream& os, const Layer& layer)
      -> std::ostream& {
    for (const auto& neuron : layer.neurons_) {
      os << neuron;
    }
    return os;
  }

 private:
  std::vector<Neuron> neurons_;
};

}  // namespace s21

#endif  // MLP_MODEL_LAYER_H_
