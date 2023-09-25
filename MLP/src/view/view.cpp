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
  auto result = controller_.StartLearn(learningfile_, ui->EpochspinBox->value(),
                                       ui->TypeBox->currentIndex());
  BuildGraph(result);
}

void view::BuildGraph(std::pair<QVector<double>, QVector<double>> data) {
  ui->plot->addGraph();
  ui->plot->graph(0)->setData(data.second, data.first);
  ui->plot->graph(0)->setLineStyle(QCPGraph::lsLine);
  ui->plot->graph(0)->setScatterStyle(QCPScatterStyle::ssCircle);
  ui->plot->xAxis->setLabel("Номер эпохи");
  ui->plot->yAxis->setLabel("Значение ошибки");
  ui->plot->xAxis->setRange(0, data.second.last());
  ui->plot->yAxis->setRange(
      0, *std::max_element(data.first.begin(), data.first.end()));
  ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
  ui->plot->replot();
}

void view::update(QImage image) {
  auto vec = NormalizeAndConvertToEigen(image);
  int index = 0;
  auto result = controller_.ForwardFeed(vec, ui->TypeBox->currentIndex());
  result.maxCoeff(&index);
  QString letter = letters_[index];
  ui->resultlabel->clear();
  ui->resultlabel->setText(letter);
}

void view::on_StartTestingButton_clicked() {
  ui->resulttest->setText(controller_.StartTest(
      testfile_, ui->SimpleRateSpinbox->value(), ui->TypeBox->currentIndex()));
}

void view::on_ImportWeightsButton_clicked() {
  QString filename = QFileDialog::getOpenFileName(this, "Open File", "~/",
                                                  "Text files (*.txt)");

  if (!filename.isEmpty())
    controller_.LoadWeights(filename.toStdString(),
                            ui->TypeBox->currentIndex());
}

void view::on_ExportWeights_clicked() {
  QString filename = QFileDialog::getSaveFileName(this, "Save File", "~/",
                                                  "Text files (*.txt)");

  if (!filename.isEmpty())
    controller_.SaveWeights(filename.toStdString(),
                            ui->TypeBox->currentIndex());
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
    auto result = controller_.ForwardFeed(vec, ui->TypeBox->currentIndex());
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
  transform.rotate(90);
  QImage rotatedImage = resizedImage.transformed(transform);
  Eigen::VectorXd vec(784);
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      QColor color(rotatedImage.pixel(x, y));
      vec[y * 28 + x] = qGray(color.rgb());
    }
  }
  return vec;
}

void view::on_HIddenLayersspinbox_valueChanged(int arg1) {
  controller_.SetLayers(arg1);
}
