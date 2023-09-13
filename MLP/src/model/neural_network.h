
#ifndef MLP_MODEL_NEURAL_NETWORK_H_
#define MLP_MODEL_NEURAL_NETWORK_H_

#include "layer.h"

namespace s21 {

class NeuralNetwork {
 public:
  NeuralNetwork(std::size_t layers, std::size_t neurons) {
    for (auto i = 0; i < layers; ++i) {
      layers_.push_back(Layer(neurons));
    }
  }
  NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork& neuralNetwork) = default;
  NeuralNetwork(NeuralNetwork&& neuralNetwork) = default;
  NeuralNetwork& operator=(const NeuralNetwork& neuralNetwork) = default;
  NeuralNetwork& operator=(NeuralNetwork&& neuralNetwork) = default;
  ~NeuralNetwork() = default;
  auto FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd& gradients, double learningRate)
      -> void;
  [[nodiscard]] auto GetLayers() const -> std::vector<Layer> { return layers_; }

  friend auto operator<<(std::ostream& os, const NeuralNetwork& neuralNetwork)
      -> std::ostream& {
    for (const auto& layer : neuralNetwork.layers_) {
      os << layer;
    }
    return os;
  }

 private:
  std::vector<Layer> layers_;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURAL_NETWORK_H_
