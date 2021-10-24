#include "qdmgraphicssocketin.h"

QDMGraphicsSocketIn::QDMGraphicsSocketIn()
{
    label->setPos(SIZE / 2, -SIZE * 0.7f);
}

void QDMGraphicsSocketIn::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPainterPath pathContent;
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    pathContent.addRoundedRect(rect, ROUND, ROUND);
    painter->setPen(Qt::NoPen);
    painter->setBrush(Qt::blue);
    painter->drawPath(pathContent.simplified());
}

void QDMGraphicsSocketIn::linkAttached(QDMGraphicsLinkFull *link)
{
    unlinkAll();
    QDMGraphicsSocket::linkAttached(link);
}
