#include "graph_perceptron.h"

namespace s21 {

GraphPerceptrone::GraphPerceptrone(std::size_t layers, std::size_t neurons,
                                   std::size_t inputs) {
  layers_.reserve(layers + 2);
  layers_.push_back(LayerGraph(neurons, inputs));
  for (std::size_t i = 0; i < layers; ++i) {
    layers_.push_back(LayerGraph(neurons, neurons));
  }
  auto last = LayerGraph(26, neurons);
  layers_.push_back(last);
}

auto GraphPerceptrone::Train(double learningrate, const Eigen::VectorXd &inputs,
                             const Eigen::VectorXd &target) -> double {
  auto result = FeedForward(inputs);
  return BackPropagation(result, target, learningrate);
}

// void GraphPerceptrone::SaveWeights(std::string filename) {
//   //   std::ofstream file(filename);
//   //   if (file.is_open()) {
//   //     file << *this;
//   //   }
// }

// void GraphPerceptrone::LoadWeights(std::string filename) {
//   //   std::ifstream file(filename);
//   //   if (file.is_open()) {
//   //     file >> *this;
//   //   }
// }

auto GraphPerceptrone::FeedForward(const Eigen::VectorXd &inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  std::for_each(layers_.begin(), layers_.end(),
                [&](auto &layer) { outputs = layer.FeedForward(outputs); });
  return outputs;
}

auto GraphPerceptrone::BackPropagation(const Eigen::VectorXd &outputnetwork,
                                       const Eigen::VectorXd &target,
                                       double learningRate) -> double {
  auto error = target - outputnetwork;

  Eigen::VectorXd errorfirst = layers_.back().BackPropagation(
      error, learningRate, layers_[layers_.size() - 2]);

  for (auto i = layers_.size() - 2; i > 0; --i) {
    errorfirst =
        layers_[i].BackPropagation(errorfirst, learningRate, layers_[i - 1]);
  }
  double mse = error.squaredNorm() / error.size();
  return mse;
}

// auto operator<<(std::ostream &os, const GraphPerceptrone &neuralNetwork)
//     -> std::ostream & {
//   //   for (const auto &layer : neuralNetwork.layers_) {
//   //     os << layer;
//   //   }
//   //   return os;
// }

// auto operator>>(std::ifstream &is, GraphPerceptrone &neuralNetwork)
//     -> std::ifstream & {
//   //   for (auto &layer : neuralNetwork.layers_) {
//   //     is >> layer;
//   //   }
//   //   return is;
// }

}  // namespace s21
