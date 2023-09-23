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

}  // namespace s21