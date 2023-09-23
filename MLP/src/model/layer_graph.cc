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
  // новая ошибка
  Eigen::VectorXd newError{layer.Size()};
  for (auto i = 0; i < layer.Size(); ++i) {
    newError[i] = 0;
    for (auto j = 0; j < nodes_.size(); ++j) {
      newError[i] += nodes_[j](i) * error[j];
    }
    newError[i] *= layer(i).Derivative();
  }

  // изменяем веса
  auto LastLayerOutput = layer.GetOutputNeurons();
  for (auto i = 0; i < nodes_.size(); ++i) {
    double gradient = error[i] * nodes_[i].Derivative();
    nodes_[i].UpdateWeights(gradient, learningRate, LastLayerOutput);
  }

  return newError;
}

}  // namespace s21