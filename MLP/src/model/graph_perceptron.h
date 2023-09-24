
#ifndef MLP_MODEL_LAYER_GRAPH_PERCEPTRONE_H_
#define MLP_MODEL_LAYER_GRAPH_PERCEPTRONE_H_

#include "layer_graph.h"

namespace s21 {

class GraphPerceptrone {
public:
  GraphPerceptrone(std::size_t layers = 3, std::size_t neurons = 300,
                   std::size_t inputs = 784);
  GraphPerceptrone(const GraphPerceptrone &GraphPerceptrone) = default;
  GraphPerceptrone(GraphPerceptrone &&GraphPerceptrone) = default;
  GraphPerceptrone &
  operator=(const GraphPerceptrone &GraphPerceptrone) = default;
  GraphPerceptrone &operator=(GraphPerceptrone &&GraphPerceptrone) = default;
  ~GraphPerceptrone() = default;

  auto FeedForward(const Eigen::VectorXd &inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd &outputnetwork,
                       const Eigen::VectorXd &, double learningRate) -> double;
  auto Train(double learningrate = 0.05,
             const Eigen::VectorXd &inputs = Eigen::VectorXd::Random(784),
             const Eigen::VectorXd &target = Eigen::VectorXd::Random(26))
      -> double;
  auto SaveWeights(std::string filename) -> void;
  auto LoadWeights(std::string filename) -> void;
  friend auto operator<<(std::ostream &os,
                         const GraphPerceptrone &GraphPerceptrone)
      -> std::ostream &;
  friend auto operator>>(std::ifstream &is, GraphPerceptrone &GraphPerceptrone)
      -> std::ifstream &;

private:
  std::vector<LayerGraph> layers_;
};

} // namespace s21

#endif // MLP_MODEL_LAYER_GRAPH_PERCEPTRONE_H_
