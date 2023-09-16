#include "controler.h"
namespace s21 {
void Controler::StartLearn() { model.StartLearn(); }

void Controler::StartTest() { model.StartTest(); }

Controler::Controler() {}
} // namespace s21
