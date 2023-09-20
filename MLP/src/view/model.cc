#include "model.h"
namespace s21 {
void Model::StartLearn() {
  auto parsedatas2 = Parse("/opt/goinfre/barnards/VladBigBrain/MLP/datasets/"
                           "emnist-letters-train.csv");
  double initial_learning_rate = 0.01;
  // Константа затухания
  double decay_constant = 0.0001;
  for (auto i = 0; i < 5; ++i) {
    double learning_rate =
        initial_learning_rate * std::exp(-decay_constant * i);
    for (auto &i : parsedatas2) {
      network_.Train(learning_rate, i.input, i.correct_vector);
    }
  }
}

void Model::StartTest() {
  int correct = 0, incorrect = 0;
  for (auto &temp : parsedatas) {
    Eigen::VectorXd result = network_.FeedForward(temp.input);
    int maxIndex = 0;
    result.maxCoeff(&maxIndex);
    int trueLabelIndex = 0;
    temp.correct_vector.maxCoeff(&trueLabelIndex);
    if (maxIndex == trueLabelIndex) {
      correct++;
    } else {
      incorrect++;
    }
  }

  // Output results (or do further calculations)
  std::cout << std::endl
            << "Correct: " << correct << ", Incorrect: " << incorrect
            << std::endl;
}

Eigen::VectorXd Model::ForwardFeed(Eigen::VectorXd input) {
  return network_.FeedForward(input);
}

std::vector<Data> Model::Parse(const std::string &filename) {
  std::vector<Data> dataset;
  io::LineReader reader(filename);
  std::vector<std::string> data;
  while (auto temp = reader.next_line()) {
    std::istringstream lineStream(temp);
    std::string cell;
    std::getline(lineStream, cell);
    data.push_back(cell);
  }
  dataset = ConvertToEigen(data);
  return dataset;
}

Model::Model() {}

std::vector<Data> Model::ConvertToEigen(const std::vector<std::string> &data) {
  std::vector<Data> dataset;

  for (const auto &full_row : data) {
    std::stringstream row_stream(full_row);
    std::string cell;
    std::getline(row_stream, cell, ',');
    int label = std::stoi(cell);
    Eigen::VectorXd labelVector = Eigen::VectorXd::Zero(26);
    labelVector[label - 1] = 1.0;
    //    std::cout << label-1 << std::endl;
    Eigen::VectorXd pixels(784);
    int i = 0;
    while (std::getline(row_stream, cell, ',')) {
      pixels[i] = std::stod(cell);
      i++;
    }

    dataset.push_back({labelVector, pixels});
  }
  return dataset;
}

} // namespace s21
