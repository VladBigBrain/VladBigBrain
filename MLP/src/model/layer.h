#ifndef MLP_MODEL_LAYER_H_
#define MLP_MODEL_LAYER_H_

#include "neuron.h"

namespace s21 {

class Layer {
 public:
  Layer(std::vector<Neuron> neurons) : neurons_(neurons) {}
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