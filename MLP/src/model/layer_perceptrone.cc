#include "layer.h"

namespace s21 {
auto Layer::FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  Eigen::VectorXd output(inputs.size());
  for (size_t i = 0; i < neurons_.size(); ++i) {
    double weighted_sum = 0.0;
    for (size_t j = 0; j < inputs.size(); ++j) {
      weighted_sum += inputs(j) * neurons_[i][j];
    }
    // weighted_sum += neurons_[i].bias;
    output(i) = neurons_[i].Activate(weighted_sum);
  }
  return output;
}
auto Layer::BackPropagation(const Eigen::VectorXd& gradients,
                            double learningRate) -> void {}
}  // namespace s21