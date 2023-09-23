#ifndef MLP_VIEW_VIEW_H
#define MLP_VIEW_VIEW_H

#include <QFileDialog>
#include <QMainWindow>

#include "controler.h"

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
  void update(QImage);
  void on_StartTestingButton_clicked();
  void on_ImportWeightsButton_clicked();
  void on_ExportWeights_clicked();
  void on_learningimportbuttonresult_clicked();
  void on_Testingimportbutton_clicked();
  void on_ImportIMageButton_clicked();

 private:
  s21::Controler controller_;
  std::map<int, QString> letters_;
  std::string learningfile_{};
  std::string testfile_{};
  Ui::view *ui;

  Eigen::VectorXd NormalizeAndConvertToEigen(const QImage &originalImage);
  void BuildGraph(std::pair<QVector<double>, QVector<double>>);
};

#endif  // MLP_VIEW_VIEW_H
