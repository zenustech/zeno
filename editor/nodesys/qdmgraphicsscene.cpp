#include "qdmgraphicsscene.h"
#include "qdmgraphicsview.h"
#include "serialization.h"
#include <zeno/ztd/memory.h>
#include <zeno/zmt/log.h>
#include <zeno/zan/zan.h>
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
    removeItem(node);
    nodes.erase(ztd::stale_ptr(node));
    emit sceneUpdated();
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

std::string QDMGraphicsScene::getName() const
{
    return subnetNode ? subnetNode->getName() : "/";
}

std::string QDMGraphicsScene::allocateNodeName(std::string const &prefix) const
{
    return zan::find_unique_name
        ( nodes
        | zan::map(ZENO_F1(p, p->getName()))
        , prefix);
}

void QDMGraphicsScene::doubleClicked()
{
    auto node = addNode();
    node->initAsSubnet();
    emit sceneCreatedOrRemoved();
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

    emit sceneUpdated();
    return link;
}

void QDMGraphicsScene::removeLink(QDMGraphicsLinkFull *link)
{
    link->srcSocket->linkRemoved(link);
    link->dstSocket->linkRemoved(link);
    removeItem(link);
    links.erase(ztd::stale_ptr(link));

    emit sceneUpdated();
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

    emit sceneUpdated();
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
    ZENO_DEBUG("copyPressed: {}", res);
}

void QDMGraphicsScene::pastePressed()
{
    ZENO_DEBUG("pastePressed");
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

std::vector<QDMGraphicsNode *> QDMGraphicsScene::getVisibleNodes() const
{
    std::vector<QDMGraphicsNode *> res;
    for (auto const &node: nodes) {
        res.push_back(node.get());
    }
    return res;
}

std::vector<QDMGraphicsScene *> QDMGraphicsScene::getChildScenes() const
{
    std::vector<QDMGraphicsScene *> res;
    for (auto const &node: nodes) {
        if (auto subnet = node->getSubnetScene())
            res.push_back(subnet);
    }
    return res;
}

ZENO_NAMESPACE_END
