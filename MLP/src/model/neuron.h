#ifndef MLP_MODEL_NEURON_H_
#define MLP_MODEL_NEURON_H_

#include <Eigen/Dense>

namespace s21 {

class Neuron {
 public:
  Neuron(double value = 0, Eigen::VectorXd weights = Eigen::VectorXd::Zero(1),
         double bias = 0)
      : value(value), weights(weights), bias(bias) {}

  Neuron(const Neuron&) = default;
  Neuron(Neuron&&) = default;
  auto operator=(const Neuron&) -> Neuron& = default;
  auto operator=(Neuron&&) -> Neuron& = default;
  ~Neuron() = default;

  auto UpdateWeights(double learningRate) -> void {
    // weights_ += learningRate * Derivative(value_);
  }

  auto Derivative(double value) -> double {
    double sigmoid = Activate(value);
    return sigmoid * (1 - sigmoid);
  }

  auto Activate(double value) -> double { return 1 / (1 + std::exp(-value)); }

  friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
    os << "VALUE " << neuron.value << " "
       << "Bias " << neuron.bias << std::endl;
    os << "WEIGHTS" << std::endl;
    for (const auto& weight : neuron.weights) {
      os << " " << weight;
    }
    os << std::endl;
    return os;
  }

 private:
  double value = 0;
  Eigen::VectorXd weights;
  double bias = 0;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURON_H_