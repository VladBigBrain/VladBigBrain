#ifndef MLP_INTERFACES_NEURAL_NETWORK_H_
#define MLP_INTERFACES_NEURAL_NETWORK_H_

#include <Eigen/Dense>

#include "layer.h"
namespace s21 {

class NeuralNetwork {
 public:
  NeuralNetwork(double learning_rate, size_t hidden_layers, size_t epochs,
                Eigen::MatrixXd data)
      : learning_rate_(learning_rate),
        hidden_layers_(hidden_layers),
        epochs_(epochs),
        data_(data){};

  NeuralNetwork(const NeuralNetwork&) = default;
  NeuralNetwork(NeuralNetwork&&) = default;
  auto operator=(const NeuralNetwork&) -> NeuralNetwork& = default;
  auto operator=(NeuralNetwork&&) -> NeuralNetwork& = default;
  virtual ~NeuralNetwork() = default;

  virtual auto Train() -> void = 0;
  virtual auto Predict(const Eigen::VectorXd& input) -> Eigen::VectorXd = 0;
  virtual auto SaveModel(const std::string& filename) -> void = 0;
  virtual auto LoadModel(const std::string& filename) -> void = 0;

 private:
  double learning_rate_;
  size_t hidden_layers_;
  size_t epochs_;
  Eigen::MatrixXd data_;
  std::vector<Layer> layers_;
};

}  // namespace s21

#endif  // MLP_INTERFACES_NEURAL_NETWORK_H_
