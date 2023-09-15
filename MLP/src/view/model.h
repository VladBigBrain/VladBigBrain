#ifndef MODEL_H
#define MODEL_H

#include "csv.h"
#include "neural_network.h"

namespace s21 {
struct Data {
  Eigen::VectorXd correct_vector;
  Eigen::VectorXd input;
  friend std::ostream &operator<<(std::ostream &os, const Data &data) {
      os << data.correct_vector;
      os << data.input;
      return os;
  }
};

class Model {
public:
  void StartLearn();
  std::vector<Data> Parse(const std::string &filename);
  Model();
  std::vector<Data> ConvertToEigen(const std::vector<std::string>& data);
private:
  NeuralNetwork network;
};
} // namespace s21
#endif // MODEL_H
