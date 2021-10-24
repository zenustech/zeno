#include "qdmgraphicslinkhalf.h"
#include "qdmgraphicssocketin.h"
#include "qdmgraphicssocketout.h"
#include "qdmgraphicsscene.h"
#include <QGraphicsView>
#include <QCursor>

QDMGraphicsLinkHalf::QDMGraphicsLinkHalf(QDMGraphicsSocket *socket)
    : socket(socket)
{
    setZValue(1);
}

QPointF QDMGraphicsLinkHalf::getSrcPos() const {
    if (auto sock = dynamic_cast<QDMGraphicsSocketOut *>(socket))
        return sock->getLinkedPos();
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    return parentScene->getCursorPos();
}

QPointF QDMGraphicsLinkHalf::getDstPos() const {
    if (auto sock = dynamic_cast<QDMGraphicsSocketIn *>(socket))
        return sock->getLinkedPos();
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    return parentScene->getCursorPos();
}
