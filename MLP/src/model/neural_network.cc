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
  for (size_t i = 0; i < epochs; ++i) {
    auto result = FeedForward(inputs);
    BackPropagation(result, target, 0.1);
  }
}

auto NeuralNetwork::FeedForward(const Eigen::VectorXd &inputs)
    -> Eigen::VectorXd {
  Eigen::VectorXd outputs = inputs;
  int i = 0;
  ForEach(layers_, [&](auto &layer) { outputs = layer.FeedForward(outputs); });
  return outputs;
}
auto NeuralNetwork::BackPropagation(const Eigen::VectorXd &inputs,
                                    const Eigen::VectorXd &target,
                                    double learningRate) -> void {
//  std::cerr << " input cols " << inputs.cols() << " Input rows "
//            << inputs.rows() << std::endl;

//  std::cerr << "i'm back propagation" << std::endl;
  // error
  auto error = target - inputs;

  // local gradient for each neuron
  Eigen::VectorXd gradient =
      error.array() * layers_.back().GetDerivativeVector().array(); // 26

  //  std::cerr << " i'm calc gradient " << std::endl;
  // calc deltaweights
  Eigen::MatrixXd deltaweights =
      gradient * layers_[layers_.size() - 2].GetOutputNeurons().transpose() *
      learningRate; // x / 26
  //  std::cerr << " Deltaweight cools " << deltaweights.cols()
  //            << " Deltaweights rows " << deltaweights.rows() << std::endl;

  //  std::cerr << "i'm deltaweights " << std::endl;
  const auto &olewights = layers_.back().GetWeights();

  //  std::cerr << "olewights weights colls" << olewights.cols() << "  "
  //            << "olewights weight rows = " << olewights.rows() << std::endl;

  // calc error output
  Eigen::VectorXd errorfirst =
      layers_.back().GetWeights().transpose() * gradient;
  //  std::cerr << "i'm errorfirst" << std::endl;
  // set new weights

  Eigen::MatrixXd newweights = olewights - deltaweights;

  //  std::cerr << "current weights colls" << newweights.cols() << "  "
  //            << "current weight rows = " << newweights.rows() << std::endl;

  layers_.back().SetWeights(newweights);

  // work with another layer
  for (auto i = layers_.size() - 2; i > 0; --i) {
//    std::cout << "i'm learning" << std::endl;
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
