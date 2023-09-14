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

TEST(Test, FeedForward) {
  s21::NeuralNetwork nn(2, 3, 5);
  auto inputs = Eigen::RowVectorXd::Random(5);
  auto result = nn.FeedForward(inputs);
  ASSERT_EQ(result.size(), 26);
}
