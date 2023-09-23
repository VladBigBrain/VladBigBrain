#ifndef MLP_MODEL_LAYER_GRAPH_H_
#define MLP_MODEL_LAYER_GRAPH_H_

#include "node.h"

namespace s21 {

class LayerGraph {
 public:
  LayerGraph(std::size_t neurons, std::size_t inputs);
  LayerGraph(const LayerGraph &) = default;
  LayerGraph(LayerGraph &&) = default;
  auto operator=(const LayerGraph &) -> LayerGraph & = default;
  auto operator=(LayerGraph &&) -> LayerGraph & = default;
  ~LayerGraph() = default;

  auto FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd;
  // auto BackPropagation(const Eigen::VectorXd &error, double learningRate,
  //                      LayerGraph &layer) -> Eigen::VectorXd;

 private:
  std::vector<Node> nodes_;
};

}  // namespace s21

#endif  // MLP_MODEL_LAYER_GRAPH_H_