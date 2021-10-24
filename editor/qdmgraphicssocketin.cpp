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
