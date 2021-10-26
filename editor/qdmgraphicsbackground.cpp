#include "qdmgraphicsbackground.h"
#include "qdmgraphicsscene.h"
#include <QPainterPath>
#include <QDebug>

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
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    parentScene->blankClicked();

    QGraphicsItem::mousePressEvent(event);
}
