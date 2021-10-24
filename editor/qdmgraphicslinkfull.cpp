#include "qdmgraphicslinkfull.h"

QDMGraphicsLinkFull::QDMGraphicsLinkFull(QDMGraphicsSocketOut *srcSocket, QDMGraphicsSocketIn *dstSocket)
    : srcSocket(srcSocket), dstSocket(dstSocket)
{
    setFlag(QGraphicsItem::ItemIsSelectable);
    srcSocket->linkAttached(this);
    dstSocket->linkAttached(this);
}

QPointF QDMGraphicsLinkFull::getSrcPos() const {
    return srcSocket->getLinkedPos();
}

QPointF QDMGraphicsLinkFull::getDstPos() const {
    return dstSocket->getLinkedPos();
}
