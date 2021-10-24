#include "qdmgraphicslinkfull.h"

QDMGraphicsLinkFull::QDMGraphicsLinkFull(QDMGraphicsSocketOut *srcSocket, QDMGraphicsSocketIn *dstSocket)
    : srcSocket(srcSocket), dstSocket(dstSocket)
{
    setFlag(QGraphicsItem::ItemIsSelectable);
    srcSocket->linkAttached(this);
    dstSocket->linkAttached(this);
}

QPointF QDMGraphicsLinkFull::getSrcPos() const {
    return srcSocket->scenePos();
}

QPointF QDMGraphicsLinkFull::getDstPos() const {
    return dstSocket->scenePos();
}
