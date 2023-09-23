#ifndef MLP_VIEW_MODEL_H
#define MLP_VIEW_MODEL_H

#include <QVector>
#include <future>
#include <iostream>
#include <vector>

#include "csv.h"
#include "graph_perceptron.h"
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
  Model() = default;
  Model(Model &) = default;
  Model(Model &&) = default;
  auto operator=(Model &) -> Model & = default;
  auto operator=(Model &&) -> Model & = default;
  ~Model() = default;

  std::pair<QVector<double>, QVector<double>> StartLearn(
      const std::string &filename, double epoch, int strategy);
  QString StartTest(const std::string &filename, float fraction, int strategy);
  Eigen::VectorXd ForwardFeed(Eigen::VectorXd input,int strategy);

  auto SaveWeights(std::string file, int strategy) -> void;
  auto LoadWeights(std::string file, int strategy) -> void;

 private:
  NeuralNetwork network_;
  GraphPerceptrone graph_network_;

  std::vector<Data> Parse(const std::string &filename);
  std::vector<Data> ConvertToEigen(const std::vector<std::string> &data);
};

}  // namespace s21

#endif  // MLP_VIEW_MODEL_H
