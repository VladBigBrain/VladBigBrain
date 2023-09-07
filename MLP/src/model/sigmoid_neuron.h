#ifndef MODEL_SIGMOID_NEURON_H_
#define MODEL_SIGMOID_NEURON_H_

#include "neuron.h"
namespace s21 {
class SigmoidNeuron : public Neuron {
 public:
  SigmoidNeuron(double value = 0, Eigen::VectorXd weights = Eigen::VectorXd(),
                double bias = 0)
      : Neuron(value, weights, bias) {}

  SigmoidNeuron(const SigmoidNeuron&) = default;
  SigmoidNeuron(SigmoidNeuron&&) = default;
  auto operator=(const SigmoidNeuron&) -> SigmoidNeuron& = default;
  auto operator=(SigmoidNeuron&&) -> SigmoidNeuron& = default;
  ~SigmoidNeuron() override = default;

  auto Derivative(double value) -> double override {
    double sigmoid = Activate(value);
    return sigmoid * (1 - sigmoid);
  }

  auto Activate(double value) -> double override {
    return 1 / (1 + std::exp(-value));
  }
};

}  // namespace s21

#endif  // MODEL_SIGMOID_NEURON_H_
