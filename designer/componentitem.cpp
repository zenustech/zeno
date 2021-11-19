#include "nodescene.h"
#include "componentitem.h"


ComponentItem::ComponentItem(NodeScene* pScene, qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent)
    : QGraphicsRectItem(0, 0, w, h, parent)
    , m_pScene(pScene)
{
    QPen pen(QColor(0, 0, 0), 3);
    pen.setJoinStyle(Qt::MiterJoin);

    setPen(pen);
    setBrush(Qt::NoBrush);
    setPos(x, y);

    setFlags(ItemIsMovable | ItemIsSelectable);

    m_pScene->addItem(this);
}

void ComponentItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
}

void ComponentItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseMoveEvent(event);
    m_pScene->updateDragPoints(this, NodeScene::TRANSLATE);
}

void ComponentItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
}

void ComponentItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    _base::paint(painter, option, widget);
}

QRectF ComponentItem::boundingRect() const
{
    QRectF br = _base::boundingRect();
    return br;
}