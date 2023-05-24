#ifndef __ZTIMESILDER_H__
#define __ZTIMESILDER_H__

#include <QtWidgets>
#include <zenoui/style/zenostyle.h>

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

    const int m_sHMargin;
    const int scaleH;
    const int smallScaleH;
    const int fontHeight;
    const int fontScaleSpacing;

    int m_from, m_value, m_to;
    QTransform m_transform;

    int m_cellLength;
    int m_lengthUnit[3] = {1, 2, 5};
    int getCellLength(int total);
};

#endif
