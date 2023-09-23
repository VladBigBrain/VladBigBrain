#include "layer_graph.h"

namespace s21 {

auto LayerGraph::FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd {
  Eigen::VectorXd result(nodes_.size());
  for (auto i = 0; i < nodes_.size(); ++i) {
    result[i] = nodes_[i].Activate(inputs);
  }
  return result;
}

LayerGraph::LayerGraph(std::size_t neurons, std::size_t inputs) {
  for (auto i = 0; i < neurons; ++i) {
    nodes_.push_back(Node(inputs));
  }
}

auto LayerGraph::Size() const -> size_t { return nodes_.size(); }

auto LayerGraph::GetDerivativeVector() -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(
      nodes_.size(), [this](Eigen::Index i) { return nodes_[i].Derivative(); });
}
auto LayerGraph::GetOutputNeurons() -> Eigen::VectorXd {
  return Eigen::VectorXd::NullaryExpr(
      nodes_.size(), [this](Eigen::Index i) { return nodes_[i].GetValue(); });
}

auto LayerGraph::BackPropagation(const Eigen::VectorXd &error,
                                 double learningRate, LayerGraph &layer)
    -> Eigen::VectorXd {

  Eigen::MatrixXd weightMatrix =
      Eigen::MatrixXd::Zero(nodes_.size(), layer.Size());

  auto LastLayerOutput = layer.GetOutputNeurons();

  for (auto i = 0; i < nodes_.size(); ++i) {
    double gradient = error[i] * nodes_[i].Derivative();
    weightMatrix.row(i) = nodes_[i].GetWeights();
    nodes_[i].UpdateWeights(gradient, learningRate, LastLayerOutput);
  }

  return weightMatrix.transpose() * error;
}

auto operator<<(std::ostream &os, const LayerGraph &layer) -> std::ostream & {
  for (const auto &node : layer.nodes_) {
    os << node;
  }
  return os;
}

auto operator>>(std::ifstream &is, LayerGraph &layer) -> std::ifstream & {
  for (auto &node : layer.nodes_) {
    is >> node;
  }
}

} // namespace s21
