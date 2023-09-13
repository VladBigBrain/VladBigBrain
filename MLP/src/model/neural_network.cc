#include "neural_network.h"
namespace s21 {
auto NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs;
  ForEach(layers_, [&](auto& layer) { outputs = layer.FeedForward(inputs); });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd& gradients,
                                    double learningRate) -> void {
  //
}
}  // namespace s21
