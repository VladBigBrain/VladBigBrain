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

auto NeuralNetwork::ErrorFunction(double result, double target) -> double {
  return pow(result - target, 2);
}

auto NeuralNetwork::Train(size_t epochs, const Eigen::VectorXd& inputs)
    -> void {
  for (size_t i = 0; i < epochs; ++i) {
    auto result = FeedForward(inputs);
    auto errors = ErrorFunction(result.maxCoeff(), inputs[0]);
    BackPropagation(result, errors, 0.1);
  }
}

auto NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  ForEach(layers_, [&](auto& layer) { outputs = layer.FeedForward(outputs); });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd& inputs,
                                    double errors, double learningRate)
    -> void {
  auto output = inputs;
  for (auto i = layers_.size() - 1; i > 0; --i) {
    output = layers_[i].BackPropagation(output, errors, learningRate);
  }
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
