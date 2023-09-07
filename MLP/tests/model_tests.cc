#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "layer_perceptrone.h"
#include "neuron.h"
#include "random.h"

TEST(Test, Neuron) {
  s21::Neuron neuron;
  s21::PerceptroneLayer perceptrone_layer(
      s21::RandomGenerator::GenerateRandomLayer(2));
  for (auto temp : perceptrone_layer.GetNeurons()) {
    Console::WriteLine(temp);
  }
  Console::WriteLine(perceptrone_layer.GetNeurons().size());
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
