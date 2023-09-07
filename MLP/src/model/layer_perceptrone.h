#ifndef MODEL_LAYER_PERCEPTRONE_H_
#define MODEL_LAYER_PERCEPTRONE_H_

#include "neuron.h"
namespace s21 {

class PerceptroneLayer {
 public:
  PerceptroneLayer(std::vector<Neuron> neurons) : neurons_(neurons) {}
  PerceptroneLayer(const PerceptroneLayer&) = delete;
  PerceptroneLayer(PerceptroneLayer&&) = delete;
  auto operator=(const PerceptroneLayer&) -> PerceptroneLayer& = delete;
  auto operator=(PerceptroneLayer&&) -> PerceptroneLayer& = delete;
  ~PerceptroneLayer() = default;
  auto FeedForward(const Eigen::VectorXd& inputs) -> Eigen::VectorXd;
  auto BackPropagation(const Eigen::VectorXd& gradients, double learningRate)
      -> void;
  [[nodiscard]] auto GetNeurons() const -> std::vector<Neuron> {
    return neurons_;
  }

 private:
  std::vector<Neuron> neurons_;
};

}  // namespace s21

#endif  // MODEL_LAYER_PERCEPTRONE_H_
