#ifndef MODEL_H
#define MODEL_H

#include "csv.h"
#include "neural_network.h"
#include <future>
#include <iostream>
#include <vector>
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
  void StartTest() ;
  Model();

private:
  std::vector<Data> Parse(const std::string &filename);
  std::vector<Data> ConvertToEigen(const std::vector<std::string> &data);
  NeuralNetwork network_;
};
} // namespace s21
#endif // MODEL_H
