#include <gtest/gtest.h>

#include "csv.h"
#include "graph_perceptron.h"
#include "neural_network.h"

TEST(Test, FeedForward) {
  s21::NeuralNetwork nn(2, 3, 5);
  auto inputs = Eigen::VectorXd::Random(5);
  auto result = nn.FeedForward(inputs);
  ASSERT_EQ(result.size(), 26);
}

TEST(Test, Back) {
  s21::NeuralNetwork nn(2, 120, 784);
  auto inputs = Eigen::VectorXd::Random(784);
  auto result = nn.FeedForward(inputs);

  auto fnc = [&](const Eigen::VectorXd& inputs, int target) -> Eigen::VectorXd {
    Eigen::VectorXd targetVector(inputs.size());
    targetVector.setZero();
    targetVector(target) = 1.0;
    return targetVector;
  };

  nn.BackPropagation(result, fnc(result, 13), 0.1);
  ASSERT_EQ(result.size(), 26);
}
