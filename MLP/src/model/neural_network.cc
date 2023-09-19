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

auto NeuralNetwork::Train(size_t epochs, const Eigen::VectorXd &inputs,
                          const Eigen::VectorXd &target) -> void {
  auto result = FeedForward(inputs);
  BackPropagation(result, target, 0.1);
}

auto NeuralNetwork::FeedForward(const Eigen::VectorXd &inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  ForEach(layers_, [&](auto &layer) { outputs = layer.FeedForward(outputs); });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd &outputnetwork,
                                    const Eigen::VectorXd &target,
                                    double learningRate) -> void {
  // error
  auto error = target - outputnetwork;

  // local gradient for each neuron
  Eigen::VectorXd gradient =
      error.array() * layers_.back().GetDerivativeVector().array(); // 26

  // calc deltaweights
  Eigen::MatrixXd deltaweights =
      gradient * layers_[layers_.size() - 2].GetOutputNeurons().transpose() *
      learningRate; // x / 26

  const Eigen::MatrixXd &old_weights = layers_.back().GetWeights();

  // calc error output
  Eigen::VectorXd errorfirst = old_weights.transpose() * gradient;

  // set new weights
  layers_.back().SetWeights(old_weights + deltaweights);

  for (auto i = layers_.size() - 2; i > 0; --i) {
    errorfirst =
        layers_[i].BackPropagation(errorfirst, learningRate, layers_[i - 1]);
  }
}

auto operator<<(std::ostream &os, const NeuralNetwork &neuralNetwork)
    -> std::ostream & {
  auto i = 0;
  for (const auto &layer : neuralNetwork.layers_) {
    std::cout << "Layer " << ++i << ":" << std::endl;
    os << layer;
  }
  std::cout << std::endl;
  return os;
}

} // namespace s21
