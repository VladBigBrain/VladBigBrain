#include "controler.h"
namespace s21 {
void Controler::StartLearn() { model.StartLearn(); }

void Controler::StartTest() { model.StartTest(); }

Eigen::VectorXd Controler::ForwardFeed(Eigen::VectorXd input) {
  return model.ForwardFeed(input);
}

void Controler::SaveWeights(std::string file) { model.SaveWeights(file); }

void Controler::LoadWeights(std::string file) { model.LoadWeights(file); }

Controler::Controler() {}
} // namespace s21
