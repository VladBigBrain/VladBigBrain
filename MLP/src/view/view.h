#ifndef VIEW_H
#define VIEW_H

#include "controler.h"
#include <QMainWindow>
QT_BEGIN_NAMESPACE
namespace Ui {
class view;
}
QT_END_NAMESPACE

class view : public QMainWindow {
  Q_OBJECT

public:
  view(QWidget *parent = nullptr);
  ~view();

private slots:
  void on_Learnbutton_clicked();

  void on_StartTestingButton_clicked();

private:
  s21::Controler controller;
  Ui::view *ui;
};
#endif // VIEW_H
