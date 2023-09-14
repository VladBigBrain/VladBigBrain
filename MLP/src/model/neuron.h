#ifndef MLP_MODEL_NEURON_H_
#define MLP_MODEL_NEURON_H_

#include <Eigen/Dense>

namespace s21 {

class Neuron {
 public:
  Neuron(double value = 0, double bias = 0);
  Neuron(const Neuron&) = default;
  Neuron(Neuron&&) = default;
  auto operator=(const Neuron&) -> Neuron& = default;
  auto operator=(Neuron&&) -> Neuron& = default;
  ~Neuron() = default;

  auto Derivative() -> double;
  auto Activate(double value) -> double;

  friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron);

 private:
  double value_ = 0;
  double bias_ = 0;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURON_H_
