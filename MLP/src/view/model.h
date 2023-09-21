#ifndef MODEL_H
#define MODEL_H

#include <future>
#include <iostream>
#include <vector>

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
  void StartTest();
  Eigen::VectorXd ForwardFeed(Eigen::VectorXd input);
  Model();
  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;

 private:
  std::vector<Data> Parse(const std::string &filename);
  std::vector<Data> ConvertToEigen(const std::vector<std::string> &data);
  std::vector<Data> parsedatas = Parse("/opt/goinfre/barnards/VladBigBrain/MLP/datasets/"
                          "emnist-letters-test.csv");
  NeuralNetwork network_;
};
}  // namespace s21
#endif  // MODEL_H
