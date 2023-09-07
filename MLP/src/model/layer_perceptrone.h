#ifndef MODEL_LAYER_PERCEPTRONE_H_
#define MODEL_LAYER_PERCEPTRONE_H_

#include "layer.h"
namespace s21 {

class PerceptroneLayer : public Layer {
 public:
  PerceptroneLayer(std::vector<Neuron> neurons) : Layer(neurons){};
  PerceptroneLayer(const PerceptroneLayer&) = default;
  PerceptroneLayer(PerceptroneLayer&&) = default;
  auto operator=(const PerceptroneLayer&) -> PerceptroneLayer& = default;
  auto operator=(PerceptroneLayer&&) -> PerceptroneLayer& = default;
  ~PerceptroneLayer() = default;
  auto FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd override;
  auto BackPropagation(const Eigen::VectorXd& gradients, double learningRate)
      -> void override;
};

}  // namespace s21

#endif  // MODEL_LAYER_PERCEPTRONE_H_
