#ifndef MLP_VIEW_MODEL_H
#define MLP_VIEW_MODEL_H

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
  Model() = default;
  Model(Model &) = default;
  Model(Model &&) = default;
  auto operator=(Model &) -> Model & = default;
  auto operator=(Model &&) -> Model & = default;
  ~Model() = default;

  std::pair<QVector<double>, QVector<double>>
  StartLearn(const std::string &filename, double epoch);
  QString StartTest(const std::string &filename, float fraction);
  Eigen::VectorXd ForwardFeed(Eigen::VectorXd input);

  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;

private:
  NeuralNetwork network_;

  std::vector<Data> Parse(const std::string &filename);
  std::vector<Data> ConvertToEigen(const std::vector<std::string> &data);
};

} // namespace s21

#endif // MLP_VIEW_MODEL_H
