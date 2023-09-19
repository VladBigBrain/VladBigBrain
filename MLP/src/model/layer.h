#ifndef MLP_MODEL_LAYER_H_
#define MLP_MODEL_LAYER_H_

#include <iostream>

#include "algorithm.h"
#include "neuron.h"
namespace s21 {

class Layer {
public:
  Layer(std::size_t neurons, std::size_t inputs);
  Layer(const Layer &) = default;
  Layer(Layer &&) = default;
  auto operator=(const Layer &) -> Layer & = default;
  auto operator=(Layer &&) -> Layer & = default;
  ~Layer() = default;
  auto FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd &error, double learningRate,
                       Layer &layer) -> Eigen::VectorXd;
  auto GetDerivativeVector() -> Eigen::VectorXd;
  auto GetWeights() -> Eigen::MatrixXd { return weights_; }
  auto GetOutputNeurons() -> Eigen::VectorXd;
  auto SetWeights(const Eigen::MatrixXd &weights) -> void;
  auto SetBias(const Eigen::VectorXd bias) -> void { bias_ = bias; }
  auto Size() const -> size_t;
  friend auto operator<<(std::ostream &os, const Layer &layer)
      -> std::ostream &;
  const Eigen::VectorXd &bias() const;

  const Eigen::MatrixXd &getPrevious_deltaweights() const;
  void setPrevious_deltaweights(const Eigen::MatrixXd &newPrevious_deltaweights);

private:
  auto BuildGradientMatrix(const Eigen::VectorXd &error) -> Eigen::MatrixXd;
  auto BuildMatrixOfWeights(const std::size_t inputs) -> void;
  auto BuildNeurons(const Eigen::VectorXd &out) -> Eigen::VectorXd;
  [[nodiscard]] auto GetNeurons() const -> std::vector<Neuron>;
  std::vector<Neuron> neurons_;
  Eigen::MatrixXd weights_;
  Eigen::MatrixXd previous_deltaweights;
  Eigen::VectorXd bias_;
};

} // namespace s21

#endif // MLP_MODEL_LAYER_H_
