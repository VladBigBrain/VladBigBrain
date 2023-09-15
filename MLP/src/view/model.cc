#include "model.h"
namespace s21 {
void Model::StartLearn() {
  auto parsedatas = Parse("/opt/goinfre/barnards/VladBigBrain/MLP/datasets/"
                          "emnist-letters-train.csv");
  std::cout << parsedatas[0];
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

    // Читаем метку
    std::getline(row_stream, cell, ',');
    int label = std::stoi(cell);
    Eigen::VectorXd labelVector = Eigen::VectorXd::Zero(27);
    labelVector[label] = 1.0; // One-hot encoding

    Eigen::VectorXd pixels(784);

    // Читаем пиксели
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
