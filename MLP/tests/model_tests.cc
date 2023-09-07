#include <gtest/gtest.h>

#include "console.h"
#include "csv.h"
#include "neuron.h"
#include "sigmoid_neuron.h"

TEST(Test, Neuron) {
  s21::SigmoidNeuron neuron;
  std::cout << neuron.Activate(2);
  std::cout << neuron.Activate(3);
}

TEST(Test, CSVParser) {
  io::LineReader reader(
      "/opt/goinfre/barnards/CPP7_MLP-0/datasets/emnist-letters-train.csv");
  std::vector<std::string> data;
  while (auto temp = reader.next_line()) {
    std::istringstream lineStream(temp);
    std::string cell;
    std::getline(lineStream, cell);
    data.push_back(cell);
  }
  Console::WriteLine(data[0]);
}