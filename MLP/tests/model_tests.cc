#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "neural_network.h"

auto fnc(Eigen::VectorXd& inputs) {
  for (auto t : inputs) {
    Console::WriteLine(t);
  }
}

TEST(Test, Constructors) {
  s21::NeuralNetwork nn(2, 3, 5);
  // Console::WriteLine(nn);
  ASSERT_EQ(nn.GetLayers().size(), 4);
}

TEST(Test, FeedForward) {
  s21::NeuralNetwork nn(2, 3, 5);
  auto inputs = Eigen::RowVectorXd::Random(5);
  // Console::WriteLine("Before");
  // Console::WriteLine(nn);
  auto result = nn.FeedForward(inputs);
  Console::WriteLine(nn);
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
