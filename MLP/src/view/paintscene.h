#ifndef PAINTSCENE_H
#define PAINTSCENE_H
#include "ui_PaintScene.h"
#include <QWidget>
#include <QtWidgets>
class PaintScene : public QWidget {
  Q_OBJECT

  bool draw;
  QVector<QPointF> vv;
  QImage pic;

public:
  PaintScene(QWidget *parent = Q_NULLPTR);
  void paintEvent(QPaintEvent *);
  void mousePressEvent(QMouseEvent *);
  void mouseMoveEvent(QMouseEvent *);
  void mouseReleaseEvent(QMouseEvent *);
  ~PaintScene();
public slots:
  void Clear();

private:
  Ui::PaintScene ui;
};

#endif // PAINTSCENE_H
