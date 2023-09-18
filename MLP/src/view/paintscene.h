#ifndef PAINTSCENE_H
#define PAINTSCENE_H
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPointF>
#include <QVector>
#include <QWidget>
#include <iostream>
class PaintScene : public QWidget {
  Q_OBJECT

 public:
  explicit PaintScene(QWidget *parent = nullptr);
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

 private:
  bool draw;
  QVector<QPointF> points;
};

#endif  // PAINTSCENE_H
