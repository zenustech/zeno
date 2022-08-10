#ifndef __ZSCALE_SLIDER_H__
#define __ZSCALE_SLIDER_H__

#include <QtWidgets>

class ZTextLabel;

class ZScaleSlider : public QWidget
{
    Q_OBJECT
public:
    ZScaleSlider(QVector<qreal> scales, QWidget* parent = nullptr);
    ~ZScaleSlider();

protected:
    void paintEvent(QPaintEvent* event);
    void keyReleaseEvent(QKeyEvent* event);
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

signals:
    void numSlided(qreal);
    void slideFinished();

private:
    QVector<qreal> m_scales;
    QPoint m_lastPos;
    bool m_currScale;
    ZTextLabel* m_currLabel;
};


#endif