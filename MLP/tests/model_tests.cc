#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "neural_network.h"
#include "random.h"
TEST(Test, Constructors) {
  s21::NeuralNetwork nn(
      s21::RandomGenerator::GenerateRandomNeuralNetwork(2, 3));
  for (auto& layer : nn.GetLayers()) {
    Console::WriteLine(layer);
  }
}

TEST(Test, CSVParser) {
  // io::LineReader reader(
  //     "../../datasets/emnist-letters-train.csv");
  // std::vector<std::string> data;
  // while (auto temp = reader.next_line()) {
  //   std::istringstream lineStream(temp);
  //   std::string cell;
  //   std::getline(lineStream, cell);
  //   data.push_back(cell);
  // }
  // Console::WriteLine(data[0]);
}
