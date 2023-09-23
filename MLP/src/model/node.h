#ifndef MLP_MODEL_NODE_H_
#define MLP_MODEL_NODE_H_

#include <Eigen/Dense>
#include <random>
namespace s21 {

class Node {
 public:
  Node(std::size_t neurons);
  ~Node() = default;
  auto Activate(Eigen::VectorXd inputs) -> double;

 private:
  double value_ = 0;
  Eigen::VectorXd weights_;
  Eigen::VectorXd bias_;
  Eigen::VectorXd velocity_;

  auto BuildOfWeights(const std::size_t neurons, const std::size_t inputs)
      -> void;
  auto Sigmoid(double value) -> double;
  auto Derivative() -> double;
  auto Summator(const Eigen::VectorXd& inputs) -> Eigen::VectorXd;
};

}  // namespace s21

#endif  // MLP_MODEL_NODE_H_
