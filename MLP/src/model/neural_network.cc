#include "neural_network.h"
namespace s21 {
auto NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  Eigen::VectorXd curr_input = inputs;
  for (Layer& layer : layers_) {
    curr_input = layer.FeedForward(curr_input);
  }
  return curr_input;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd& gradients,
                                    double learningRate) -> void {
  //
}
}  // namespace s21
