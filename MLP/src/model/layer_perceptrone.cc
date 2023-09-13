#include "layer.h"

namespace s21 {
auto Layer::FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  Eigen::MatrixXd weights = BuildMatrixOfWeights();
  Eigen::VectorXd output = inputs * weights;
  BuildNeurons(output);
  return output;
}

auto Layer::BackPropagation(const Eigen::VectorXd& gradients,
                            double learningRate) -> void {}
}  // namespace s21