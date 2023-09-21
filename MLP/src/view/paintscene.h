#ifndef MLP_SRC_VIEW_SCRIBBLE_WIDGET_H_
#define MLP_SRC_VIEW_SCRIBBLE_WIDGET_H_

#include <QMouseEvent>
#include <QPainter>
#include <QWidget>

class PaintScene : public QWidget {
  Q_OBJECT

 public:
  PaintScene(QWidget* parent = nullptr);
  ~PaintScene() = default;
  PaintScene(const PaintScene&) = delete;
  PaintScene(PaintScene&&) = delete;
  PaintScene& operator=(const PaintScene&) = delete;
  PaintScene& operator=(PaintScene&&) = delete;

  bool OpenImage(const QString& filename);

 signals:
  void updated(QImage);
  void cleared();

 protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void paintEvent(QPaintEvent* event) override;
 private:
  bool scribbling_ = false;
  QPoint start_point_;
  QImage image_;
  QPainter line_painter_, image_painter_;
  const int kPenWidth_ = 60;
  const int kImageWidth_ = 512, kImageHeight_ = 512;

  void DrawLineTo(const QPoint& end_point);
};

#endif  // MLP_SRC_VIEW_SCRIBBLE_WIDGET_H_
