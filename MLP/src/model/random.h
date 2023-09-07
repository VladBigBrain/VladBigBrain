#ifndef MLP_LIBRARIES_RANDOM_H_
#define MLP_LIBRARIES_RANDOM_H_
#include <Eigen/Dense>

#include "neuron.h"

namespace s21 {

class RandomGenerator {
 public:
  static auto GenerateRandomWeights(int size) -> Eigen::VectorXd {
    Eigen::VectorXd weights = Eigen::VectorXd::Random(size);
    return weights;
  }

  static auto GenerateRandomLayer(int size) -> std::vector<Neuron> {
    std::vector<Neuron> neurons;
    for (int i = 0; i < size; i++) {
      neurons.push_back(Neuron(0, GenerateRandomWeights(size), 0));
    }
    return neurons;
  }
};

}  // namespace s21
#endif  // MLP_LIBRARIES_RANDOM_H_
