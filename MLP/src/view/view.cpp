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

  for (int i = 0; i < 26; ++i) {
    char letter = 'A' + i; // Calculate the letter based on its position
    letters_[i] = QString(1, letter); // Add the letter to the map
  }
  connect(ui->paintwidget, SIGNAL(updated(QImage)), this, SLOT(update(QImage)));
}

view::~view() { delete ui; }

void view::on_Learnbutton_clicked() { controller.StartLearn(); }

void view::update(QImage image) {
  QImage resizedImage =
      image.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
//  resizedImage.bits();
  // Step 2: Initialize Eigen Vector
  Eigen::VectorXd vec(784); // 28x28 = 784

  // Step 3: Fill Vector
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      QColor color(resizedImage.pixel(x, y));
      int grayValue = qGray(color.rgb());
      vec[y * 28 + x] = grayValue == 255 ? 0 : 1; // 0 for white, 1 for black
    }
  }
}

void view::on_StartTestingButton_clicked() { controller.StartTest(); }
