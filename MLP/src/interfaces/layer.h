#ifndef MLP_INTERFACES_LAYER_H_
#define MLP_INTERFACES_LAYER_H_

#include <vector>

#include "neuron.h"

namespace s21 {

class Layer {
 public:
  Layer(std::vector<Neuron> neurons) : neurons_(neurons){};

  Layer(const Layer&) = default;
  Layer(Layer&&) = default;
  auto operator=(const Layer&) -> Layer& = default;
  auto operator=(Layer&&) -> Layer& = default;
  ~Layer() = default;
  
  auto Begin() -> std::vector<Neuron>::iterator{return neurons_.begin();}
  auto End() -> std::vector<Neuron>::iterator{return neurons_.end();}
  
  virtual auto FeedForward(const Eigen::VectorXd& inputs)
      -> Eigen::VectorXd = 0;
  virtual auto BackPropagation(const Eigen::VectorXd& gradients,
                               double learningRate) -> void = 0;
  [[nodiscard]] auto GetNeurons() const -> const std::vector<Neuron>& {
    return neurons_;
  }

  auto SetNeurons(const std::vector<Neuron>& neurons) -> void {
    neurons_ = neurons;
  }

 private:
  std::vector<Neuron> neurons_;
};

}  // namespace s21

#endif  // MLP_INTERFACES_LAYER_H_
