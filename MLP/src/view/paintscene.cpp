#include "paintscene.h"
#include "QtWidgets/qgraphicseffect.h"
#include <QPainter>

PaintScene::PaintScene(QWidget *parent) : QWidget(parent), draw(false) {

  QGraphicsDropShadowEffect *effect = new QGraphicsDropShadowEffect();
  setFixedSize(329, 428); // или любой другой размер
}

void PaintScene::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.fillRect(event->rect(), Qt::white);
  painter.setPen(QPen(Qt::black, 20)); // Черный цвет, толщина 5
  for (const auto &point : points) {
    painter.drawPoint(point);
  }
}

void PaintScene::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    draw = true;
    points.push_back(event->pos());
    update();
  } else if (event->button() == Qt::RightButton) {
    points.clear();
    update();
  }
}

void PaintScene::mouseMoveEvent(QMouseEvent *event) {
  if (draw) {
    points.push_back(event->pos());
    update();
  }
}

void PaintScene::mouseReleaseEvent(QMouseEvent *event) { draw = false; }
