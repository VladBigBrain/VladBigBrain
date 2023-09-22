
#ifndef MLP_MODEL_NEURAL_NETWORK_H_
#define MLP_MODEL_NEURAL_NETWORK_H_

#include "layer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

namespace s21 {

class NeuralNetwork {
public:
  //    144 3
  NeuralNetwork(std::size_t layers = 3, std::size_t neurons = 300,
                std::size_t inputs = 784);
  NeuralNetwork(const NeuralNetwork &neuralNetwork) = default;
  NeuralNetwork(NeuralNetwork &&neuralNetwork) = default;
  NeuralNetwork &operator=(const NeuralNetwork &neuralNetwork) = default;
  NeuralNetwork &operator=(NeuralNetwork &&neuralNetwork) = default;
  ~NeuralNetwork() = default;

  auto FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd &outputnetwork,
                       const Eigen::VectorXd &, double learningRate) -> double;
  auto Train(double learningrate = 0.05,
             const Eigen::VectorXd &inputs = Eigen::VectorXd::Random(784),
             const Eigen::VectorXd &target = Eigen::VectorXd::Random(26))
      -> double;
  auto SaveWeights(std::string filename) -> void;
  auto LoadWeights(std::string filename) -> void;
  friend auto operator<<(std::ostream &os, const NeuralNetwork &neuralNetwork)
      -> std::ostream &;
  friend auto operator>>(std::ifstream &is, NeuralNetwork &neuralNetwork)
      -> std::ifstream &;

private:
  std::vector<Layer> layers_;
};

} // namespace s21

#endif // MLP_MODEL_NEURAL_NETWORK_H_
