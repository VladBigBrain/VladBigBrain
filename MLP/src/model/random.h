#ifndef MLP_LIBRARIES_RANDOM_H_
#define MLP_LIBRARIES_RANDOM_H_
#include <Eigen/Dense>

#include "neural_network.h"

namespace s21 {

class RandomGenerator {
 public:
  static auto GenerateRandomWeights(int size) -> Eigen::VectorXd {
    Eigen::VectorXd weights = Eigen::VectorXd::Random(size);
    return weights;
  }

  static auto GenerateRandomLayer(int count_layers, int count_neurons)
      -> std::vector<Neuron> {
    std::vector<Neuron> neurons;
    for (int i = 0; i < count_layers; i++) {
      neurons.push_back(Neuron(0, GenerateRandomWeights(count_neurons), 0));
    }
    return neurons;
  }

  static auto GenerateRandomNeuralNetwork(int count_layers, int count_neurons)
      -> std::vector<Layer> {
    std::vector<Layer> layers;
    for (int i = 0; i < count_layers; i++) {
      layers.push_back(GenerateRandomLayer(count_layers, count_neurons));
    }
    return layers;
  }
};

}  // namespace s21
#endif  // MLP_LIBRARIES_RANDOM_H_
