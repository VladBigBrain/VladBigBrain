#include "controler.h"

namespace s21 {

std::pair<QVector<double>, QVector<double>>
Controler::StartLearn(const std::string &filename, double epoch, int strategy) {
  return model_.StartLearn(filename, epoch, strategy);
}

QString Controler::StartTest(const std::string &filename, float fraction,
                             int strategy) {
  return model_.StartTest(filename, fraction, strategy);
}

Eigen::VectorXd Controler::ForwardFeed(Eigen::VectorXd input, int strategy) {
  return model_.ForwardFeed(input, strategy);
}

void Controler::SaveWeights(std::string file, int strategy) {
  model_.SaveWeights(file, strategy);
}

void Controler::LoadWeights(std::string file, int strategy) {
  model_.LoadWeights(file, strategy);
}

} // namespace s21
