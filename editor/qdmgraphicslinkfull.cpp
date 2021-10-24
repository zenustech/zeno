#include "qdmgraphicslinkfull.h"
#include "qdmgraphicsscene.h"

QDMGraphicsLinkFull::QDMGraphicsLinkFull(QDMGraphicsSocketOut *srcSocket, QDMGraphicsSocketIn *dstSocket)
    : srcSocket(srcSocket), dstSocket(dstSocket)
{
    setZValue(-1);
    setFlag(QGraphicsItem::ItemIsSelectable);
    srcSocket->linkAttached(this);
    dstSocket->linkAttached(this);
}

QPointF QDMGraphicsLinkFull::getSrcPos() const
{
    return srcSocket->getLinkedPos();
}

QPointF QDMGraphicsLinkFull::getDstPos() const
{
    return dstSocket->getLinkedPos();
}

void QDMGraphicsLinkFull::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        auto parentScene = static_cast<QDMGraphicsScene *>(scene());
        parentScene->removeLink(this);
        return;
    }

    QGraphicsItem::mousePressEvent(event);
}
