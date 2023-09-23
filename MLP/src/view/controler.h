#ifndef MLP_VIEW_CONTROLER_H
#define MLP_VIEW_CONTROLER_H

#include "model.h"

namespace s21 {

class Controler {
 public:
  Controler() = default;
  Controler(Controler &) = default;
  Controler(Controler &&) = default;
  auto operator=(Controler &) -> Controler & = default;
  auto operator=(Controler &&) -> Controler & = default;
  ~Controler() = default;

  auto StartLearn(const std::string &filename, double epoch, int strategy)
      -> std::pair<QVector<double>, QVector<double>>;
  auto StartTest(const std::string &filename, float fraction, int strategy)
      -> QString;
  auto ForwardFeed(Eigen::VectorXd input) -> Eigen::VectorXd;

  auto SaveWeights(std::string file) -> void;
  auto LoadWeights(std::string file) -> void;

 private:
  Model model_;
};

}  // namespace s21

#endif  // MLP_VIEW_CONTROLER_H
