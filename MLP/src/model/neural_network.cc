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

// auto NeuralNetwork::ErrorFunction(const Eigen::VectorXd &inputs, int target)
//     -> Eigen::VectorXd {
//   Eigen::VectorXd targetVector(inputs.size());
//   targetVector.setZero();
//   targetVector(target) = 1.0;
//   return targetVector;
// }

auto NeuralNetwork::Train(size_t epochs, const Eigen::VectorXd &inputs,
                          const Eigen::VectorXd &target) -> void {
  for (size_t i = 0; i < epochs; ++i) {
    auto result = FeedForward(inputs);
    BackPropagation(result, target, 0.1);
  }
}

auto NeuralNetwork::FeedForward(const Eigen::VectorXd &inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  int i = 0;
  ForEach(layers_, [&](auto &layer) {
    // std::cout << "i = " << i++ << std::endl;
    outputs = layer.FeedForward(outputs);
  });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd &inputs,
                                    const Eigen::VectorXd &target,
                                    double learningRate) -> void {
  // error
  auto error = target - inputs;

  // local gradient for each neuron
  Eigen::VectorXd gradient =
      error.array() * layers_.back().GetDerivativeVector().array();  // 26

  // calc deltaweights
  auto deltaweights =
      learningRate * gradient *
      layers_[layers_.size() - 2].GetOutputNeurons().transpose();  // x / 26

  // calc error output
  Eigen::VectorXd errorfirst =
      layers_.back().GetWeights().transpose() * gradient;

  // set new weights
  layers_.back().SetWeights(layers_.back().GetWeights() - deltaweights);

  // work with another layer
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

}  // namespace s21
