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
  // nodes_.reserve(neurons + 2);
  // nodes_.push_back(Node(neurons, inputs));
  // for (std::size_t i = 0; i < neurons; ++i) {
  //   nodes_.push_back(Node(neurons, neurons));
  // }
  // auto last = Node(neurons, neurons);
  // nodes_.push_back(last);
}

}  // namespace s21