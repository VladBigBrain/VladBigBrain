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
  auto Derivative() -> double;
  auto UpdateWeights(double gradient = 0, double learningRate = 0,
                     const Eigen::VectorXd& output = Eigen::VectorXd()) -> void;
  auto GetValue() const -> double { return value_; }
  auto operator()(size_t i) -> double& { return weights_[i]; }

 private:
  double value_ = 0;
  Eigen::VectorXd weights_;
  Eigen::VectorXd bias_;
  Eigen::VectorXd velocity_;
  auto BuildOfWeights(const std::size_t neurons) -> void;
  auto Sigmoid(double value) -> double;
};

}  // namespace s21

#endif  // MLP_MODEL_NODE_H_
