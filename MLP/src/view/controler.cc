#include "controler.h"

namespace s21 {

std::pair<QVector<double>, QVector<double>>
Controler::StartLearn(const std::string &filename, double epoch) {
  return model_.StartLearn(filename, epoch);
}

QString Controler::StartTest(const std::string &filename, float fraction) {
  return model_.StartTest(filename, fraction);
}

Eigen::VectorXd Controler::ForwardFeed(Eigen::VectorXd input) {
  return model_.ForwardFeed(input);
}

void Controler::SaveWeights(std::string file) { model_.SaveWeights(file); }

void Controler::LoadWeights(std::string file) { model_.LoadWeights(file); }

} // namespace s21
