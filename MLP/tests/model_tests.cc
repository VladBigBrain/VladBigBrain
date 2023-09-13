#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "neural_network.h"
#include "random.h"

auto fnc(Eigen::VectorXd& inputs) {
  for (auto t : inputs) {
    Console::WriteLine(t);
  }
}

TEST(Test, Constructors) {
  s21::NeuralNetwork nn(
      s21::RandomGenerator::GenerateRandomNeuralNetwork(2, 3));
  for (auto i : nn.GetLayers()) {
    Console::WriteLine(i);
  }
  ASSERT_EQ(nn.GetLayers().size(), 2);
}

TEST(Test, FeedForward) {
  s21::NeuralNetwork nn(
      s21::RandomGenerator::GenerateRandomNeuralNetwork(2, 3));
  auto inputs = Eigen::RowVectorXd::Random(3);
  // Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 2);
  // Eigen::VectorXd x = Eigen::VectorXd::Random(2);
  // Eigen::VectorXd y = A * x;

  auto result = nn.FeedForward(inputs);
  // for (auto t : result) {
  //   Console::WriteLine(t);
  // }
}

// TEST(Test, CSVParser) {
//   // io::LineReader reader(
//   //     "../../datasets/emnist-letters-train.csv");
//   // std::vector<std::string> data;
//   // while (auto temp = reader.next_line()) {
//   //   std::istringstream lineStream(temp);
//   //   std::string cell;
//   //   std::getline(lineStream, cell);
//   //   data.push_back(cell);
//   // }
//   // Console::WriteLine(data[0]);
// }
