#ifndef __ZTIMESILDER_H__
#define __ZTIMESILDER_H__

#include <QtWidgets>

class ZSlider : public QWidget
{
    Q_OBJECT
public:
    ZSlider(QWidget* parent = nullptr);
    virtual QSize sizeHint() const override;
    void setFromTo(int from, int to);
    int value() const;

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

signals:
    int sliderValueChange(int);

public slots:
    void setSliderValue(int value);

private:
    int _posToFrame(int x);
    int _frameToPos(int frame);
    int _getframes();
    void drawSlideHandle(QPainter* painter, int scaleH);
    const int m_sHMargin = 13;
    const int scaleH = 9;
    const int smallScaleH = 5;
    const int fontHeight = 15;  //QFont font("Segoe UI", 9);
    const int fontScaleSpacing = 4;

    int m_from, m_value, m_to;
    QTransform m_transform;
};

#endif
