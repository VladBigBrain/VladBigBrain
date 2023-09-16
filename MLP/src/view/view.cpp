#include "view.h"
#include "./ui_view.h"
#include "QtWidgets/qgraphicseffect.h"

void applyShadowEffectToGroupBox(QGroupBox *box) {
  QGraphicsDropShadowEffect *shadowEffect = new QGraphicsDropShadowEffect(box);
  shadowEffect->setOffset(5, 5);
  shadowEffect->setBlurRadius(15);
  box->setGraphicsEffect(shadowEffect);
}

view::view(QWidget *parent) : QMainWindow(parent), ui(new Ui::view) {
  ui->setupUi(this);
}

view::~view() { delete ui; }

void view::on_Learnbutton_clicked() { controller.StartLearn(); }

void view::on_StartTestingButton_clicked() { controller.StartTest(); }
