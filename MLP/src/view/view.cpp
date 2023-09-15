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
  QList<QGroupBox *> allGroupBoxes = this->findChildren<QGroupBox *>();
  for (QGroupBox *box : allGroupBoxes) {
    applyShadowEffectToGroupBox(box);
  }
}

view::~view() { delete ui; }

void view::on_Learnbutton_clicked() {}
