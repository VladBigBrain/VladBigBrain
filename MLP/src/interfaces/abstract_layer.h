
#ifndef MLP_INTERFACES_ABSTRACT_LAYER_H_
#define MLP_INTERFACES_ABSTRACT_LAYER_H_

#include "abstract_neuron.h"

#include <Eigen/Dense>
#include <vector>

namespace s21 {

class AbstractLayer {
public:
  virtual ~AbstractLayer() = default;
  
  virtual Eigen::VectorXd FeedForward(const Eigen::VectorXd &inputs) = 0;
  virtual Eigen::VectorXd BackPropagation(const Eigen::VectorXd &error, double learningRate) = 0;
  virtual Eigen::VectorXd GetDerivativeVector() = 0;

protected:
  std::vector<AbstractNeuron> neurons_;
};

} // namespace s21

#endif  // MLP_INTERFACES_ABSTRACT_LAYER_H_
