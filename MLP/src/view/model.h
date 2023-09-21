#ifndef MODEL_H
#define MODEL_H

#include "csv.h"
#include "neural_network.h"
#include <QVector>
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
  std::pair<QVector<double>, QVector<double>>
  StartLearn(const std::string &filename, double epoch);
  QString StartTest(const std::string &filename, float fraction);
  Eigen::VectorXd ForwardFeed(Eigen::VectorXd input);
  Model();
  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;

private:
  std::vector<Data> Parse(const std::string &filename);
  std::vector<Data> ConvertToEigen(const std::vector<std::string> &data);
  NeuralNetwork network_;
};
} // namespace s21
#endif // MODEL_H
