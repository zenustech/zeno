#include "qdmgraphicssocketin.h"

QDMGraphicsSocketIn::QDMGraphicsSocketIn()
{
    label->setPos(SIZE / 2, -SIZE * 0.7f);
}

void QDMGraphicsSocketIn::linkAttached(QDMGraphicsLinkFull *link)
{
    unlinkAll();
    QDMGraphicsSocket::linkAttached(link);
}

QPointF QDMGraphicsSocketIn::getLinkedPos() const
{
    return scenePos() - QPointF(SIZE / 2, 0);
}
