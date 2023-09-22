
#ifndef MLP_INTERFACES_ABSTRACT_NEURON_H_
#define MLP_INTERFACES_ABSTRACT_NEURON_H_


#include <Eigen/Dense>
#include <iostream>

namespace s21 {

class AbstractNeuron {
public:
  virtual ~AbstractNeuron() = default;  


  virtual double GetValue() const = 0;
  virtual double Derivative() = 0;
  virtual double Activate(double value) = 0;

  friend std::ostream &operator<<(std::ostream &os, const AbstractNeuron &neuron);

protected:
  double value_ = 0;
};

}  // namespace s21

#endif  // MLP_INTERFACES_ABSTRACT_NEURON_H_
