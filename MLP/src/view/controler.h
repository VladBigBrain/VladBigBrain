#ifndef CONTROLER_H
#define CONTROLER_H
#include "model.h"
namespace s21 {
class Controler {
public:
  auto StartLearn(const std::string &filename) -> void;
  auto StartTest(const std::string &filename) -> void;
  auto ForwardFeed(Eigen::VectorXd input) -> Eigen::VectorXd;
  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;
  Controler();

private:
  Model model;
};
} // namespace s21
#endif // CONTROLER_H
