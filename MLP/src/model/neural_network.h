
#ifndef MLP_MODEL_NEURAL_NETWORK_H_
#define MLP_MODEL_NEURAL_NETWORK_H_

#include "layer.h"

namespace s21 {

class NeuralNetwork {
 public:
  NeuralNetwork(const std::vector<Layer>& layers) : layers_(layers) {}
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

 private:
  std::vector<Layer> layers_;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURAL_NETWORK_H_
