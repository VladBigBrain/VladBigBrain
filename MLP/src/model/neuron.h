#ifndef MLP_MODEL_NEURON_H_
#define MLP_MODEL_NEURON_H_

#include <Eigen/Dense>

namespace s21 {

class Neuron {
 public:
  Neuron(double value = 0, std::size_t weights = 0, double bias = 0)
      : value_(value), bias_(bias) {
    weights_ = Eigen::VectorXd::Random(weights);
  }

  Neuron(const Neuron&) = default;
  Neuron(Neuron&&) = default;
  auto operator=(const Neuron&) -> Neuron& = default;
  auto operator=(Neuron&&) -> Neuron& = default;
  ~Neuron() = default;

  auto UpdateWeights(double learningRate, double delta) -> void {
    // weights = weights + (learningRate * delta);
  }

  auto Derivative(double value) -> double {
    double sigmoid = Activate(value);
    return sigmoid * (1 - sigmoid);
  }

  [[nodiscard]] auto GetWeights() -> Eigen::VectorXd { return weights_; }

  auto Activate(double value) -> double { return 1 / (1 + std::exp(-value)); }

  auto operator[](size_t index) -> double& { return weights_(index); }
  auto operator()(double value) -> void { value_ = Activate(value); }
  
  friend std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
    os << "VALUE " << neuron.value_ << " "
       << "Bias " << neuron.bias_ << std::endl;
    os << "WEIGHTS" << std::endl;
    for (const auto& weight : neuron.weights_) {
      os << " " << weight;
    }
    os << std::endl;
    return os;
  }

 private:
  double value_ = 0;
  Eigen::VectorXd weights_;
  double bias_ = 0;
};

}  // namespace s21

#endif  // MLP_MODEL_NEURON_H_
