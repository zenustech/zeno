#ifndef __ZTIMESILDER_H__
#define __ZTIMESILDER_H__

#include <QtWidgets>

class _QSlider : public QSlider
{
public:
    _QSlider(QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* e);
};

class ZSlider : public QWidget
{
    Q_OBJECT
public:
    ZSlider(QWidget* parent = nullptr);
    virtual QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event);
    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);

signals:
    int sliderValueChange(int);

public slots:
    void setSliderValue(int value);

private:
    int _posToFrame(int x);
    int _frameToPos(int frame);
    int _getframes();
    void drawSlideHandle(QPainter* painter);
    const int m_sHMargin = 13;

    int m_left;
    int m_right;
    int m_value;
    QTransform m_transform;
};

#endif
