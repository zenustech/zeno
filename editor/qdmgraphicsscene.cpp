#include "qdmgraphicsscene.h"
#include "qdmgraphicsview.h"
#include <zeno/ztd/memory.h>

QDMGraphicsScene::QDMGraphicsScene()
{
    float w = 100000, h = 100000;
    QRectF rect(-w / 2, -h / 2, w, h);
    setSceneRect(rect);

    background = std::make_unique<QDMGraphicsBackground>();
    addItem(background.get());
}

QDMGraphicsScene::~QDMGraphicsScene() = default;

void QDMGraphicsScene::removeNode(QDMGraphicsNode *node)
{
    node->unlinkAll();
    removeItem(node);
    nodes.erase(zeno::ztd::stale_unique_ptr(node));
    delete node;
}

void QDMGraphicsScene::socketClicked(QDMGraphicsSocket *socket)
{
    if (!pendingLink) {
        pendingLink = std::make_unique<QDMGraphicsLinkHalf>(socket);
        addItem(pendingLink.get());
    } else {
        removeItem(pendingLink.get());
        addLink(socket, pendingLink->socket);
        pendingLink = nullptr;
    }
}

void QDMGraphicsScene::blankClicked()
{
    if (floatingNode) {
        floatingNode = nullptr;
    }

    if (pendingLink) {
        pendingLink->socket->linkAttached(nullptr);
        removeItem(pendingLink.get());
        pendingLink = nullptr;
   }
}

void QDMGraphicsScene::cursorMoved()
{
    if (floatingNode) {
        floatingNode->setPos(getCursorPos());
    }

    if (pendingLink) {
        pendingLink->update();
    }
}

QDMGraphicsLinkFull *QDMGraphicsScene::addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket)
{
    auto srcIn = dynamic_cast<QDMGraphicsSocketIn *>(srcSocket);
    auto dstIn = dynamic_cast<QDMGraphicsSocketIn *>(dstSocket);
    auto srcOut = dynamic_cast<QDMGraphicsSocketOut *>(srcSocket);
    auto dstOut = dynamic_cast<QDMGraphicsSocketOut *>(dstSocket);
    std::unique_ptr<QDMGraphicsLinkFull> link;
    if (srcOut && dstIn)
        link = std::make_unique<QDMGraphicsLinkFull>(srcOut, dstIn);
    else if (dstOut && srcIn)
        link = std::make_unique<QDMGraphicsLinkFull>(dstOut, srcIn);
    else
        return nullptr;
    addItem(link.get());
    auto pLink = link.get();
    links.insert(move(link));
    return pLink;
}

void QDMGraphicsScene::removeLink(QDMGraphicsLinkFull *link)
{
    link->srcSocket->linkRemoved(link);
    link->dstSocket->linkRemoved(link);
    removeItem(link);
    links.erase(zeno::ztd::stale_unique_ptr(link));
    delete link;
}

QDMGraphicsNode *QDMGraphicsScene::addNode()
{
    auto node = std::make_unique<QDMGraphicsNode>();
    auto nodeP = node.get();
    addItem(nodeP);
    nodes.insert(move(node));
    return nodeP;
}

void QDMGraphicsScene::addNodeByName(QString name)
{
    if (floatingNode)
        return;
    auto node = addNode();
    node->initByName(name);
    node->setPos(sceneRect().bottomRight());
    floatingNode = node;
}

QPointF QDMGraphicsScene::getCursorPos() const
{
    auto view = views().at(0);
    return view->mapToScene(view->mapFromGlobal(QCursor::pos()));
}

void QDMGraphicsScene::deletePressed()
{
    std::vector<QDMGraphicsNode *> nodes;
    std::vector<QDMGraphicsLinkFull *> links;

    foreach (auto item, selectedItems()) {
        if (auto node = dynamic_cast<QDMGraphicsNode *>(item)) {
            nodes.push_back(node);
        } else if (auto link = dynamic_cast<QDMGraphicsLinkFull *>(item)) {
            links.push_back(link);
        }
    }

    for (auto link: links) {
        removeLink(link);
    }
    for (auto node: nodes) {
        removeNode(node);
    }
}
