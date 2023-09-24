#include "model.h"

namespace s21 {

std::pair<QVector<double>, QVector<double>>
Model::StartLearn(const std::string &filename, double epoch, int strategy) {
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
      if (strategy == 0) {
        currenterror = network_.Train(learning_rate, i.input, i.correct_vector);
      } else {
        currenterror =
            graph_network_.Train(learning_rate, i.input, i.correct_vector);
      }
    }
    errors.push_back(currenterror);
  }
  return {errors, epochs};
}

QString Model::StartTest(const std::string &filename, float fraction,
                         int strategy) {
  auto start_time = std::chrono::high_resolution_clock::now();
  auto testdatas_ = Parse(filename);
  int total = static_cast<int>(testdatas_.size() * fraction);
  int correct = 0, incorrect = 0, truePos = 0, falsePos = 0, falseNeg = 0;
  for (int i = 0; i < total; ++i) {
    Eigen::VectorXd result;
    if (strategy == 0) {
      result = network_.FeedForward(testdatas_[i].input);
    } else {
      result = graph_network_.FeedForward(testdatas_[i].input);
    }
    int maxIndex = 0;
    result.maxCoeff(&maxIndex);
    int trueLabelIndex = 0;
    testdatas_[i].correct_vector.maxCoeff(&trueLabelIndex);
    if (maxIndex == trueLabelIndex) {
      correct++;
      truePos++;
    } else {
      incorrect++;
      if (maxIndex != trueLabelIndex)
        falsePos++;
      if (maxIndex == trueLabelIndex)
        falseNeg++;
    }
  }
  float accuracy = static_cast<float>(correct) / total;
  float precision = static_cast<float>(truePos) / (truePos + falsePos);
  float recall = static_cast<float>(truePos) / (truePos + falseNeg);
  float f_measure = 2 * (precision * recall) / (precision + recall);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();

  return QString("Average Accuracy: %1\n"
                 "Precision: %2\n"
                 "Recall: %3\n"
                 "F-measure: %4\n"
                 "Time taken: %5 ms")
      .arg(accuracy)
      .arg(precision)
      .arg(recall)
      .arg(f_measure)
      .arg(duration);
  ;
}

Eigen::VectorXd Model::ForwardFeed(Eigen::VectorXd input, int strategy) {

  return strategy == 0 ? network_.FeedForward(input)
                       : graph_network_.FeedForward(input);
}

void Model::SetLayers(std::size_t layers_)
{
  network_ = NeuralNetwork(layers_, 300, 784);
  graph_network_ = GraphPerceptrone(layers_, 300, 784);
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

void Model::SaveWeights(std::string file, int strategy) {
  if (strategy == 0) {
    network_.SaveWeights(file);
  } else {
    graph_network_.SaveWeights(file);
  }
}

void Model::LoadWeights(std::string file, int strategy) {
  if (strategy == 0) {
    network_.LoadWeights(file);
  } else {
    graph_network_.LoadWeights(file);
  }
}

std::vector<Data> Model::ConvertToEigen(const std::vector<std::string> &data) {
  std::vector<Data> dataset;

  for (const auto &full_row : data) {
    std::stringstream row_stream(full_row);
    std::string cell;
    std::getline(row_stream, cell, ',');
    int label = std::stoi(cell);
    Eigen::VectorXd labelVector = Eigen::VectorXd::Zero(26);
    labelVector[label - 1] = 1.0;
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
