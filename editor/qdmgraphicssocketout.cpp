#include "qdmgraphicssocketout.h"

QDMGraphicsSocketOut::QDMGraphicsSocketOut()
{

}

void QDMGraphicsSocketOut::paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget)
{
    QPainterPath pathContent;
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    pathContent.addRoundedRect(rect, ROUND, ROUND);
    painter->setPen(Qt::NoPen);
    painter->setBrush(Qt::blue);
    painter->drawPath(pathContent.simplified());
}
