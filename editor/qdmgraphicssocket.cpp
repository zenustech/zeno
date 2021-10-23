#include "qdmgraphicssocket.h"
#include "qdmgraphicsnode.h"
#include "qdmgraphicsscene.h"

QDMGraphicsSocket::QDMGraphicsSocket()
{
    label = new QGraphicsTextItem(this);
}

void QDMGraphicsSocket::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton) {
        auto parentScene = static_cast<QDMGraphicsScene *>(scene());
        parentScene->socketClicked(this);
    }

    QGraphicsItem::mousePressEvent(event);
}

QRectF QDMGraphicsSocket::boundingRect() const
{
    QRectF rect(-SIZE / 2, -SIZE / 2, SIZE, SIZE);
    //auto parentNode = static_cast<QDMGraphicsNode *>(parentItem());
    //return QRectF(0, 0, parentNode->boundingRect().width(), SIZE);
    return rect;
}

void QDMGraphicsSocket::unlinkAll()
{
    auto parentScene = static_cast<QDMGraphicsScene *>(scene());
    auto saved_links = links;
    for (auto *link: saved_links) {
        parentScene->removeLink(link);
    }
}

void QDMGraphicsSocket::linkAttached(QDMGraphicsLinkFull *link)
{
    links.insert(link);
}

void QDMGraphicsSocket::linkRemoved(QDMGraphicsLinkFull *link)
{
    links.erase(link);
}

void QDMGraphicsSocket::setName(QString name)
{
    label->setPlainText(name);
}
