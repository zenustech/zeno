#include "qdmgraphicsbackground.h"
#include "qdmgraphicsscene.h"
#include <QPainterPath>

ZENO_NAMESPACE_BEGIN

QDMGraphicsBackground::QDMGraphicsBackground()
{
    setZValue(-100);
}

QRectF QDMGraphicsBackground::boundingRect() const
{
    return scene()->sceneRect();
}

void QDMGraphicsBackground::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
}

void QDMGraphicsBackground::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        auto parentScene = static_cast<QDMGraphicsScene *>(scene());
        parentScene->blankClicked();
    }

    QGraphicsItem::mousePressEvent(event);
}

void QDMGraphicsBackground::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        auto parentScene = static_cast<QDMGraphicsScene *>(scene());
        parentScene->doubleClicked();
    }

    QGraphicsItem::mousePressEvent(event);
}

ZENO_NAMESPACE_END
