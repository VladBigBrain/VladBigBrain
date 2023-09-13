#include "neural_network.h"
namespace s21 {

NeuralNetwork::NeuralNetwork(std::size_t layers, std::size_t neurons,
                             std::size_t inputs) {
  layers_.reserve(layers + 2);
  layers_.push_back(Layer(neurons, inputs));
  for (std::size_t i = 0; i < layers; ++i) {
    layers_.push_back(Layer(neurons, neurons));
  }
  auto last = Layer(26, neurons);
  layers_.push_back(last);
}

auto NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  ForEach(layers_, [&](auto& layer) { outputs = layer.FeedForward(outputs); });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd& gradients,
                                    double learningRate) -> void {
  //
}

auto operator<<(std::ostream& os, const NeuralNetwork& neuralNetwork)
    -> std::ostream& {
  auto i = 0;
  for (const auto& layer : neuralNetwork.layers_) {
    std::cout << "Layer " << ++i << ":" << std::endl;
    os << layer;
  }
  std::cout << std::endl;
  return os;
}

}  // namespace s21
