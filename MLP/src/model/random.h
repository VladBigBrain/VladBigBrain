#ifndef MLP_LIBRARIES_RANDOM_H_
#define MLP_LIBRARIES_RANDOM_H_
#include <Eigen/Dense>
#include <random>

#include "neural_network.h"

namespace s21 {

class RandomGenerator {
 public:
  static auto GenerateRandomWeights(int size) -> Eigen::VectorXd {
    Eigen::VectorXd weights = Eigen::VectorXd::Random(size);
    return weights;
  }

  static auto GenerateRandomNumber() -> double {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
  }

  static auto GenerateRandomLayer(int count_layers, int count_neurons)
      -> std::vector<Neuron> {
    std::vector<Neuron> neurons;
    for (int i = 0; i < count_layers; i++) {
      neurons.push_back(Neuron(GenerateRandomNumber(),
                               GenerateRandomWeights(count_neurons), 0));
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
