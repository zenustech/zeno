#ifndef __ZGRAPHICS_NUM_SLIDER_ITEM_H__
#define __ZGRAPHICS_NUM_SLIDER_ITEM_H__

#include <QtWidgets>

class ZSimpleTextItem;

class ZGraphicsNumSliderItem : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;

public:
    ZGraphicsNumSliderItem(const QVector<qreal>& steps, QGraphicsItem* parent = nullptr);
    ~ZGraphicsNumSliderItem();
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    void keyPressEvent(QKeyEvent* event);
    void keyReleaseEvent(QKeyEvent* event);
    void focusOutEvent(QFocusEvent* event);
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget /* = nullptr */);

signals:
    void numSlided(qreal);
    void slideFinished();

private:
    QPointF m_lastPos;
    QVector<qreal> m_steps;
    QVector<ZSimpleTextItem*> m_labels;
};

#endif