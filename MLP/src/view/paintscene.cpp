#include "paintscene.h"
#include "QtWidgets/qgraphicseffect.h"

#include <QPainter>
PaintScene::PaintScene(QWidget *parent) : QWidget(parent) {
  image_ = QImage(kImageWidth_, kImageHeight_, QImage::Format_RGB32);
  image_.fill(Qt::white);
}

bool PaintScene::OpenImage(const QString &filename) {
  QImage loaded_image;
  if (!loaded_image.load(filename))
    return false;

  image_ = loaded_image.scaled(kImageWidth_, kImageHeight_);
  update();
  emit updated(image_);
  return true;
}

void PaintScene::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    start_point_ = event->position().toPoint();
    scribbling_ = true;
  }

  if (event->button() == Qt::RightButton) {
    image_.fill(Qt::white);
    update();
    emit cleared();
  }
}

void PaintScene::mouseMoveEvent(QMouseEvent *event) {
  if (scribbling_)
    DrawLineTo(event->position().toPoint());
}

void PaintScene::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton)
    scribbling_ = false;
}

void PaintScene::paintEvent(QPaintEvent *event) {
  image_painter_.begin(this);
  QRect rect = event->rect();
  image_painter_.drawImage(rect, image_, rect);
  image_painter_.end();
}

void PaintScene::DrawLineTo(const QPoint &end_point) {
  line_painter_.begin(&image_);
  line_painter_.setRenderHint(QPainter::SmoothPixmapTransform, true);
  line_painter_.setRenderHint(QPainter::Antialiasing, true);
  line_painter_.setPen(
      QPen(Qt::black, kPenWidth_, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  line_painter_.drawLine(start_point_, end_point);
  line_painter_.end();

  start_point_ = end_point;
  update();
  emit updated(image_);
}
