#include "model.h"
namespace s21 {
void Model::StartLearn() {
  auto parsedatas = Parse("/opt/goinfre/barnards/VladBigBrain/MLP/datasets/"
                          "emnist-letters-train.csv");

  for (auto i : parsedatas) {
    network_.Train(1, i.input, i.correct_vector);
  }
}

void Model::StartTest()
{

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

    Eigen::VectorXd pixels(784);

    int i = 0;
    while (std::getline(row_stream, cell, ',')) {
      pixels[i] = std::stod(cell);
      i++;
    }

    dataset.push_back({pixels, labelVector});
  }
  return dataset;
}

} // namespace s21
