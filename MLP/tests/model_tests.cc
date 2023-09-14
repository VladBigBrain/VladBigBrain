#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "neural_network.h"

TEST(Test, Constructors) {
  s21::NeuralNetwork nn(2, 3, 5);
  ASSERT_EQ(nn.GetLayers().size(), 4);
}

TEST(Test, FeedForward) {
  s21::NeuralNetwork nn(2, 3, 5);
  auto inputs = Eigen::RowVectorXd::Random(5);
  auto result = nn.FeedForward(inputs);
  ASSERT_EQ(result.size(), 26);
}

TEST(Test, Back) {
  s21::NeuralNetwork nn(2, 3, 5);
  auto inputs = Eigen::RowVectorXd::Random(5);
  auto result = nn.FeedForward(inputs);
  auto errors = std::pow((result.maxCoeff() - inputs[0]), 2);
  nn.BackPropagation(result, errors, 0.1);
  ASSERT_EQ(result.size(), 26);
}
