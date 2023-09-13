#include "console.h"
#include "layer.h"
namespace s21 {
auto Layer::FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd {
  Eigen::MatrixXd weights = BuildMatrixOfWeights();
  Console::WriteLine("Weights");
  Console::WriteLine(weights);

  Console::WriteLine("Inputs");
  for (auto t : inputs) {
    Console::WriteLine(t);
  }

  std::cout << weights.cols() << std::endl;
  std::cout << inputs.rows() << std::endl;

  Eigen::VectorXd output = weights * inputs;
  BuildNeurons(output);
  return output;
}

auto Layer::BackPropagation(const Eigen::VectorXd& gradients,
                            double learningRate) -> void {}
}  // namespace s21