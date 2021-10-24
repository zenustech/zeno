#include "qdmgraphicsscene.h"

QDMGraphicsScene::QDMGraphicsScene()
{
    float w = 100000, h = 100000;
    QRectF rect(-w / 2, -h / 2, w, h);
    setSceneRect(rect);
}

QDMGraphicsScene::~QDMGraphicsScene()
{
    for (auto p: nodes) {
        delete p;
    }
    for (auto p: links) {
        delete p;
    }
    delete pendingLink;
}

void QDMGraphicsScene::socketClicked(QDMGraphicsSocket *socket)
{
    if (!pendingLink) {
        pendingLink = new QDMGraphicsLinkHalf(socket);
        addItem(pendingLink);
    } else {
        removeItem(pendingLink);
        addLink(socket, pendingLink->socket);
        delete pendingLink;
        pendingLink = nullptr;
    }
}

void QDMGraphicsScene::blankClicked()
{
    if (pendingLink) {
        pendingLink->socket->linkAttached(nullptr);
        removeItem(pendingLink);
        delete pendingLink;
        pendingLink = nullptr;
   }
}

void QDMGraphicsScene::cursorMoved()
{
    if (pendingLink) {
        pendingLink->update();
    }
}

void QDMGraphicsScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->buttons() & Qt::RightButton) {
        blankClicked();  // todo: use a QDMGraphicsBackground to recv mouse event instead
    }

    QGraphicsScene::mousePressEvent(event);
}

QDMGraphicsLinkFull *QDMGraphicsScene::addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket)
{
    auto srcIn = dynamic_cast<QDMGraphicsSocketIn *>(srcSocket);
    auto dstIn = dynamic_cast<QDMGraphicsSocketIn *>(dstSocket);
    auto srcOut = dynamic_cast<QDMGraphicsSocketOut *>(srcSocket);
    auto dstOut = dynamic_cast<QDMGraphicsSocketOut *>(dstSocket);
    QDMGraphicsLinkFull *link;
    if (srcOut && dstIn)
        link = new QDMGraphicsLinkFull(srcOut, dstIn);
    else if (dstOut && srcIn)
        link = new QDMGraphicsLinkFull(dstOut, srcIn);
    else
        return nullptr;
    links.insert(link);
    addItem(link);
    return link;
}

void QDMGraphicsScene::removeLink(QDMGraphicsLinkFull *link)
{
    link->srcSocket->linkRemoved(link);
    link->dstSocket->linkRemoved(link);
    removeItem(link);
    links.erase(link);
    delete link;
}

QDMGraphicsNode *QDMGraphicsScene::addNode()
{
    auto node = new QDMGraphicsNode;
    nodes.insert(node);
    addItem(node);
    return node;
}

QDMGraphicsNode *QDMGraphicsScene::addNodeByName(QString name)
{
    auto node = addNode();
    node->setupByName(name);
    return node;
}
