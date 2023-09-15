#include "model.h"
namespace s21 {
void Model::StartLearn() {
  auto parsedatas = Parse("/opt/goinfre/barnards/VladBigBrain/MLP/datasets/"
                          "emnist-letters-train.csv");
  std::cout << parsedatas[0];
}

std::vector<Data> Model::Parse(const std::string &filename) {
  //  io::CSVReader<1> in(
  //      filename); // Настроим парсер только для одного столбца (метки)
  //  int label;
  //  std::vector<Data> dataset;

  //  while (in.read_row(label)) { // Считываем только первый элемент (метку)
  //    Eigen::VectorXd pixels(784); // Вектор для пикселей
  //    Eigen::VectorXd labelVector =
  //        Eigen::VectorXd::Zero(10); // One-hot encoding вектор
  //    labelVector[label] = 1.0; // Задаем правильный ответ в one-hot encoding

  ////    for (int i = 0; i < 784; ++i) {
  ////      double pixel_value;
  ////      in.read_row(i + 1, pixel_value); // Считываем каждый следующий
  ///столбец /                                          // в переменную
  ///pixel_value /      pixels[i] = pixel_value; // Заполняем вектор пикселей /
  ///}

  //    dataset.push_back({pixels, labelVector}); // Добавляем в датасет
  //  }

  //  return dataset; // Возвращаем результат
  return {};
}

Model::Model() {}
} // namespace s21
