#ifndef MODEL_H
#define MODEL_H

#include "neural_network.h"

namespace s21 {
class Model {
public:
  void StartLearn();
  Model();

private:
  NeuralNetwork network;
};
} // namespace s21
#endif // MODEL_H
