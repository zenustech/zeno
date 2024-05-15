#ifndef __ZSCALE_SLIDER_H__
#define __ZSCALE_SLIDER_H__

#include <QtWidgets>

class ZTextLabel;

class ZNumSlider : public QWidget
{
    Q_OBJECT
public:
    ZNumSlider(QVector<qreal> scales, QWidget* parent = nullptr);
    ~ZNumSlider();

protected:
    void paintEvent(QPaintEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

signals:
    void numSlided(qreal);
    void slideFinished();

private:
    bool activateLabel(QPoint& pos, bool pressed = false);

    QVector<qreal> m_scales;
    QPoint m_lastPos;
    bool m_currScale;
    ZTextLabel* m_currLabel;
    QVector<ZTextLabel*> m_labels;
};


#endif
