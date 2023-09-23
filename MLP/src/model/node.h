#ifndef MLP_MODEL_NODE_H_
#define MLP_MODEL_NODE_H_

#include <Eigen/Dense>
#include <fstream>
#include <random>
namespace s21 {

class Node {
public:
  Node(std::size_t neurons);
  ~Node() = default;
  auto Activate(Eigen::VectorXd inputs) -> double;
  auto Derivative() -> double;
  auto UpdateWeights(double gradient = 0, double learningRate = 0,
                     const Eigen::VectorXd &output = Eigen::VectorXd()) -> void;
  auto GetValue() const -> double;
  auto GetWeights() const -> const Eigen::VectorXd &;
  auto SetWeights(Eigen::VectorXd weights) -> void;
  friend auto operator<<(std::ostream &os, const Node &node)
      -> std::ostream &;
  friend auto operator>>(std::ifstream &is, Node &node)
      -> std::ifstream &;
private:
  double value_ = 0;
  Eigen::VectorXd weights_;
  Eigen::VectorXd bias_;
  Eigen::VectorXd velocity_;
  auto BuildOfWeights(const std::size_t neurons) -> void;
  auto Sigmoid(double value) -> double;
};

} // namespace s21

#endif // MLP_MODEL_NODE_H_
