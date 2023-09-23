
#ifndef MLP_INTERFACES_ABSTRACT_NEURAL_NETWORK_H_
#define MLP_INTERFACES_ABSTRACT_NEURAL_NETWORK_H_


#include <Eigen/Dense>
#include <vector>
#include <string>


#include "abstract_layer.h" 

namespace s21 {

class AbstractNeuralNetwork {
public:
  virtual ~AbstractNeuralNetwork() = default;

  virtual Eigen::VectorXd FeedForward(const Eigen::VectorXd &inputs) = 0;
  virtual double BackPropagation(const Eigen::VectorXd &outputnetwork,
                                 const Eigen::VectorXd &target, double learningRate) = 0;
  virtual double Train(double learningRate,
                       const Eigen::VectorXd &inputs,
                       const Eigen::VectorXd &target) = 0;
  virtual void SaveWeights(std::string filename) = 0;
  virtual void LoadWeights(std::string filename) = 0;
  


protected:
  std::vector<AbstractLayer> layers_; 
};

} // namespace s21

#endif  // MLP_INTERFACES_ABSTRACT_NEURAL_NETWORK_H_

