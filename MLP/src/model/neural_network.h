
#ifndef MLP_MODEL_NEURAL_NETWORK_H_
#define MLP_MODEL_NEURAL_NETWORK_H_

#include <cmath>

#include "layer.h"

namespace s21 {

class NeuralNetwork {
 public:
  NeuralNetwork(std::size_t layers = 2, std::size_t neurons = 120,
                std::size_t inputs = 784);
  NeuralNetwork(const NeuralNetwork &neuralNetwork) = default;
  NeuralNetwork(NeuralNetwork &&neuralNetwork) = default;
  NeuralNetwork &operator=(const NeuralNetwork &neuralNetwork) = default;
  NeuralNetwork &operator=(NeuralNetwork &&neuralNetwork) = default;
  ~NeuralNetwork() = default;

  auto FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd &outputnetwork, const Eigen::VectorXd &,
                       double learningRate) -> void;
  auto Train(size_t epochs = 1,
             const Eigen::VectorXd &inputs = Eigen::VectorXd::Random(784),
             const Eigen::VectorXd &target = Eigen::VectorXd::Random(26))
      -> void;
  [[nodiscard]] auto GetLayers() const -> std::vector<Layer> { return layers_; }

  friend auto operator<<(std::ostream &os, const NeuralNetwork &neuralNetwork)
      -> std::ostream &;

 private:
  std::vector<Layer> layers_;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURAL_NETWORK_H_
