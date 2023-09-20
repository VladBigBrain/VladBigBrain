#include "controler.h"
namespace s21 {
void Controler::StartLearn() { model.StartLearn(); }

void Controler::StartTest() { model.StartTest(); }

Eigen::VectorXd Controler::ForwardFeed(Eigen::VectorXd input) {
  return model.ForwardFeed(input);
}

Controler::Controler() {}
} // namespace s21
