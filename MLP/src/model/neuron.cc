
#include "neuron.h"
namespace s21 {

Neuron::Neuron(double value, double bias) : value_(value), bias_(bias) {}

auto Neuron::Derivative() -> double {
  double sigmoid = Activate(value_);
  return sigmoid * (1 - sigmoid);
}

std::ostream& operator<<(std::ostream& os, const Neuron& neuron) {
  os << "VALUE " << neuron.value_ << " "
     << "Bias " << neuron.bias_ << std::endl;
  os << "WEIGHTS" << std::endl;
  os << std::endl;
  return os;
}

auto Neuron::Activate(double value) -> double {
  value_ = 1 / (1 + std::exp(-value));
  return value_;
}

}  // namespace s21
