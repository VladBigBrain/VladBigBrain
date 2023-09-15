#ifndef CONTROLER_H
#define CONTROLER_H
#include "model.h"
namespace s21 {
class Controler {
public:
  auto StartLearn() -> void;
  Controler();

private:
  Model model;
};
} // namespace s21
#endif // CONTROLER_H
