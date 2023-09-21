#ifndef CONTROLER_H
#define CONTROLER_H
#include "model.h"
namespace s21 {
class Controler {
public:
  auto StartLearn(const std::string &filename, double epoch)
      -> std::pair<QVector<double>, QVector<double>>;
  auto StartTest(const std::string &filename, float fraction) -> QString;
  auto ForwardFeed(Eigen::VectorXd input) -> Eigen::VectorXd;
  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;
  Controler();

private:
  Model model;
};
} // namespace s21
#endif // CONTROLER_H
