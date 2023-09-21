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
    char letter = 'A' + i;
    letters_[i] = QString(1, letter);
  }
  connect(ui->paintwidget, SIGNAL(updated(QImage)), this, SLOT(update(QImage)));
}

view::~view() { delete ui; }

void view::on_Learnbutton_clicked() {
  controller_.StartLearn(learningfile_, ui->EpochspinBox->value());
}

void view::update(QImage image) {
  auto vec = NormalizeAndConvertToEigen(image);
  //  std::cout << "Vec size: " << vec.size() << " Vec sum: " << vec.sum()
  //            << std::endl;
  int index = 0;
  auto result = controller_.ForwardFeed(vec);
  result.maxCoeff(&index);
  QString letter = letters_[index];
  //  std::cout << "Max coefficient index: " << index << std::endl;
  //  std::cerr << "LEtter" << letter.toStdString();
  ui->resultlabel->clear();
  ui->resultlabel->setText(letter);
}

void view::on_StartTestingButton_clicked() { controller_.StartTest(testfile_); }

void view::on_ImportWeightsButton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "~/",
                                                  "Text files (*.txt)");

  if (!filename.isEmpty())
    controller_.LoadWeights(filename.toStdString());
}

void view::on_ExportWeights_clicked() {
  QString filename = QFileDialog::getSaveFileName(this, "Save File", "~/",
                                                  "Text files (*.txt)");

  if (!filename.isEmpty())
    controller_.SaveWeights(filename.toStdString());
}

void view::on_learningimportbuttonresult_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "~/",
                                                  "Text files (*.csv)");
  ui->Learnbutton->setEnabled(true);
  if (!filename.isEmpty())
    learningfile_ = filename.toStdString();
}

void view::on_Testingimportbutton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "~/",
                                                  "Text files (*.csv)");
  ui->StartTestingButton->setEnabled(true);
  if (!filename.isEmpty())
    testfile_ = filename.toStdString();
}

void view::on_ImportIMageButton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "~/",
                                                  "Image files (*.bmp)");
  QImage image;
  if (image.load(filename)) {
    auto vec = NormalizeAndConvertToEigen(image);
    auto result = controller_.ForwardFeed(vec);
    int index = 0;
    result.maxCoeff(&index);
    auto letter = letters_[index];
    ui->resultlabel->setText(letter);
  } else {
    ui->resultlabel->setText("ERROR LOAD");
  }
}

Eigen::VectorXd view::NormalizeAndConvertToEigen(const QImage &originalImage) {
  QImage resizedImage = originalImage.scaled(28, 28, Qt::IgnoreAspectRatio,
                                             Qt::SmoothTransformation);

    resizedImage.invertPixels(QImage::InvertRgba);

  QTransform transform;
  transform.rotate(-90); // Поворот на 90 градусов по часовой стрелке
  QImage rotatedImage = resizedImage.transformed(transform);

  Eigen::VectorXd vec(784);

  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      QColor color(resizedImage.pixel(x, y));
      vec[y * 28 + x] = qGray(color.rgb()) / 255.0;
    }
  }

  return vec;
}
