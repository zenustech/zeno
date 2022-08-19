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
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget /* = nullptr */) override;

signals:
    void numSlided(qreal);
    void slideFinished();

private:
    QPointF m_lastPos;
    QVector<qreal> m_steps;
    QVector<ZSimpleTextItem*> m_labels;
};

#endif
