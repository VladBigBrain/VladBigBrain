#include "controler.h"
namespace s21 {
std::pair<QVector<double>, QVector<double>>
Controler::StartLearn(const std::string &filename, double epoch) {
  return model.StartLearn(filename, epoch);
}

void Controler::StartTest(const std::string &filename) {
  model.StartTest(filename);
}

Eigen::VectorXd Controler::ForwardFeed(Eigen::VectorXd input) {
  return model.ForwardFeed(input);
}

void Controler::SaveWeights(std::string file) { model.SaveWeights(file); }

void Controler::LoadWeights(std::string file) { model.LoadWeights(file); }

Controler::Controler() {}
} // namespace s21
