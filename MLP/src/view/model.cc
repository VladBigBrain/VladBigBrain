#include "model.h"
namespace s21 {
std::pair<QVector<double>, QVector<double>>
Model::StartLearn(const std::string &filename, double epoch) {
  QVector<double> errors(epoch);
  QVector<double> epochs(epoch);
  auto learningdatas_ = Parse(filename);
  double initial_learning_rate = 0.01;
  // Константа затухания
  double decay_constant = 0.0001;
  for (auto i = 1; i <= epoch; ++i) {
    epochs.push_back(i);
    double learning_rate =
        initial_learning_rate * std::exp(-decay_constant * i);
    double currenterror = 0.;
    for (auto &i : learningdatas_) {
      currenterror = network_.Train(learning_rate, i.input, i.correct_vector);
    }
    errors.push_back(currenterror);
  }
  return {errors, epochs};
}

void Model::StartTest(const std::string &filename) {
  auto testdatas_ = Parse(filename);
  int correct = 0, incorrect = 0;
  for (auto &temp : testdatas_) {
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

void Model::SaveWeights(std::string file) { network_.SaveWeights(file); }

void Model::LoadWeights(std::string file) { network_.LoadWeights(file); }

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
