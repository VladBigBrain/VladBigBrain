#ifndef MLP_INTERFACES_NEURON_H_
#define MLP_INTERFACES_NEURON_H_

#include <Eigen/Dense>
namespace s21 {
  
class Neuron {
 public:
  Neuron(double value = 0, Eigen::VectorXd weights = Eigen::VectorXd(),
         double bias = 0)
      : weights_(weights), value_(value), bias_(bias) {}

  Neuron(const Neuron&) = default;
  Neuron(Neuron&&) = default;
  auto operator=(const Neuron&) -> Neuron& = default;
  auto operator=(Neuron&&) -> Neuron& = default;
  virtual ~Neuron() = default;

  virtual auto Derivative(double value) -> double = 0;
  virtual auto Activate(double value) -> double = 0;
  
  [[nodiscard]] auto GetWeights() const -> const Eigen::VectorXd& {
    return weights_;
  }
  [[nodiscard]] auto GetBias() const -> double { return bias_; }
  [[nodiscard]] auto GetValue() const -> double { return value_; }
  auto SetWeights(const Eigen::VectorXd& weights) -> void {
    weights_ = weights;
  }
  auto SetBias(double bias) -> void { bias_ = bias; }
  auto SetValue(double value) -> void { value_ = value; }

 private:
  Eigen::VectorXd weights_;
  double value_{};
  double bias_{};
};

}  // namespace s21

#endif  // MLP_INTERFACES_NEURON_H_
