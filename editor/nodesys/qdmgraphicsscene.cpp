#include "qdmgraphicsscene.h"
#include "qdmgraphicsview.h"
#include "serialization.h"
#include <zeno/ztd/memory.h>
#include <zeno/zmt/log.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

ZENO_NAMESPACE_BEGIN

QDMGraphicsScene::QDMGraphicsScene()
{
    connect(this, SIGNAL(selectionChanged()),
            this, SLOT(updateSceneSelection()));

    float w = 100000, h = 100000;
    QRectF rect(-w / 2, -h / 2, w, h);
    setSceneRect(rect);

    background = std::make_unique<QDMGraphicsBackground>();
    addItem(background.get());
}

QDMGraphicsScene::~QDMGraphicsScene() = default;

void QDMGraphicsScene::updateSceneSelection()
{
    auto items = selectedItems();
    if (!items.size()) {
        setCurrentNode(nullptr);
        return;
    }
    if (auto node = dynamic_cast<QDMGraphicsNode *>(items.at(items.size() - 1))) {
        setCurrentNode(node);
    } else {
        setCurrentNode(nullptr);
    }
}

void QDMGraphicsScene::setCurrentNode(QDMGraphicsNode *node)
{
    currentNode = node;
    emit currentNodeChanged(node);
}

void QDMGraphicsScene::removeNode(QDMGraphicsNode *node)
{
    node->unlinkAll();
    emit nodeUpdated(node, -1);
    removeItem(node);
    nodes.erase(ztd::stale_ptr(node));
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
        floatingNode->show();
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
    QDMGraphicsLinkFull *link;
    if (srcOut && dstIn)
        link = new QDMGraphicsLinkFull(srcOut, dstIn);
    else if (dstOut && srcIn)
        link = new QDMGraphicsLinkFull(dstOut, srcIn);
    else
        return nullptr;
    addItem(link);
    links.emplace(link);

    auto dstNode = static_cast<QDMGraphicsNode *>(link->dstSocket->parentItem());
    dstNode->invalidate();
    emit nodeUpdated(dstNode, 0);
    return link;
}

void QDMGraphicsScene::removeLink(QDMGraphicsLinkFull *link)
{
    auto dstNode = static_cast<QDMGraphicsNode *>(link->dstSocket->parentItem());

    link->srcSocket->linkRemoved(link);
    link->dstSocket->linkRemoved(link);
    removeItem(link);
    links.erase(ztd::stale_ptr(link));

    dstNode->invalidate();
    emit nodeUpdated(dstNode, 0);
}

QDMGraphicsNode *QDMGraphicsScene::addNode()
{
    auto node = new QDMGraphicsNode;
    addItem(node);
    nodes.emplace(node);
    return node;
}

void QDMGraphicsScene::addNodeByType(QString type)
{
    if (floatingNode)
        return;
    auto node = addNode();
    node->initByType(type);
    node->hide();
    floatingNode = node;

    emit nodeUpdated(node, 1);
}

QPointF QDMGraphicsScene::getCursorPos() const
{
    [[unlikely]] if (!views().size())
        return this->sceneRect().topLeft();
    auto view = static_cast<QDMGraphicsView const *>(views().at(0));
    return view->mapToScene(view->mapFromGlobal(QCursor::pos()));
}

void QDMGraphicsScene::copyPressed()
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

    rapidjson::Document doc;
    serializeGraph(doc, doc.GetAllocator(), nodes, links);
    rapidjson::StringBuffer sb;
    rapidjson::Writer wr(sb);
    doc.Accept(wr);
    std::string res = sb.GetString();
    ZENO_INFO("copyPressed: {}", res);
}

void QDMGraphicsScene::pastePressed()
{
    ZENO_INFO("pastePressed");
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

ZENO_NAMESPACE_END
