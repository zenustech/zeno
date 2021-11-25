#include "qdmgraphicsscene.h"
#include "qdmgraphicsview.h"
#include "qdmgraphicsnodesubnet.h"
#include "serialization.h"
#include "utilities.h"
#include <zeno/ztd/memory.h>
#include <zeno/ztd/algorithm.h>
#include <zeno/zmt/log.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

ZENO_NAMESPACE_BEGIN

QDMGraphicsScene::QDMGraphicsScene(QDMGraphicsNodeSubnet *subnetNode)
    : subnetNode(subnetNode)
{
    connect(this, SIGNAL(selectionChanged()),
            this, SLOT(updateSceneSelection()));

    float w = 100000, h = 100000;
    QRectF rect(-w / 2, -h / 2, w, h);
    setSceneRect(rect);

    addItem(new QDMGraphicsBackground);
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
    emit currentNodeChanged(node);
}

void QDMGraphicsScene::removeNode(QDMGraphicsNode *node)
{
    for (auto *s: node->getSocketIns()) {
        removeSocketLinks(s);
    }
    for (auto *s: node->getSocketOuts()) {
        removeSocketLinks(s);
    }
    removeItem(node);
    nodes.erase(node);
    delete node;
    emit sceneUpdated();
}

void QDMGraphicsScene::socketClicked(QDMGraphicsSocket *socket)
{
    if (!pendingLink) {
        pendingLink = new QDMGraphicsLinkHalf(socket);
        addItem(pendingLink);
    } else {
        auto *toSocket = pendingLink->socket;
        removeItem(pendingLink);
        addLink(socket, toSocket);
        delete pendingLink;
        pendingLink = nullptr;

        emit sceneUpdated();
    }
}

void QDMGraphicsScene::removeSocketLinks(QDMGraphicsSocket *socket)
{
    auto links = socket->links;
    for (auto *l: links) {
        removeLink(l);
    }
}

void QDMGraphicsScene::blankClicked()
{
    if (floatingNode) {
        floatingNode->setPos(getCursorPos());
        floatingNode->show();
        floatingNode = nullptr;
    }

    if (pendingLink) {
        removeSocketLinks(pendingLink->socket);
        removeItem(pendingLink);
        delete pendingLink;
        pendingLink = nullptr;

        emit sceneUpdated();
   }
}

void QDMGraphicsScene::doubleClicked()
{
    if (subnetNode) {
        emit subnetSceneEntered(subnetNode->getScene());
    }
}

std::string QDMGraphicsScene::getName() const
{
    return subnetNode ? subnetNode->getName() : "/";
}

std::string QDMGraphicsScene::getFullPath() const
{
    if (!subnetNode)
        return "/";
    return subnetNode->getScene()->getFullPath() + '/' + subnetNode->getName();
}

std::string QDMGraphicsScene::allocateNodeName(std::string const &prefix) const
{
    std::vector<std::string> names;
    for (auto *node: nodes) {
        names.push_back(node->getName());
    }
    return find_unique_name(names, prefix);
}

void QDMGraphicsScene::addSubnetNode()
{
    if (floatingNode)
        return;
    auto node = new QDMGraphicsNodeSubnet;
    addNode(node);
    node->initialize();
    setFloatingNode(node);

    emit sceneCreatedOrRemoved();
    emit sceneUpdated();
}

void QDMGraphicsScene::setFloatingNode(QDMGraphicsNode *node)
{
    node->hide();
    node->setPos(getCursorPos());
    floatingNode = node;

    emit sceneUpdated();
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
    auto srcOut = static_cast<QDMGraphicsSocketOut *>(srcSocket);
    auto dstOut = static_cast<QDMGraphicsSocketOut *>(dstSocket);

    QDMGraphicsSocketOut *out;
    QDMGraphicsSocketIn *in;
    if (!srcIn && dstIn)
        out = srcOut, in = dstIn;
    else if (!dstIn && srcIn)
        out = dstOut, in = srcIn;
    else
        return nullptr;

    removeSocketLinks(in);
    auto link = new QDMGraphicsLinkFull(out, in);
    addItem(link);
    links.emplace(link);
    return link;
}

void QDMGraphicsScene::removeLink(QDMGraphicsLinkFull *link)
{
    link->srcSocket->links.erase(link);
    link->dstSocket->links.erase(link);
    removeItem(link);
    links.erase(link);
    delete link;

    emit sceneUpdated();
}

void QDMGraphicsScene::addNormalNode(std::string const &type)
{
    if (floatingNode)
        return;
    floatingNode = new QDMGraphicsNode;
    addNode(floatingNode);
    floatingNode->initByType(type);
    floatingNode->hide();

    emit sceneUpdated();
}

void QDMGraphicsScene::addNodeByType(QString type)
{
    auto typ = type.toStdString();
    static constexpr std::array table = {"SubnetNode"};
    switch (ztd::try_find_index(table, typ)) {
    case 0: addSubnetNode(); break;
    default: addNormalNode(typ);
    }
}

QPointF QDMGraphicsScene::getCursorPos() const
{
    if (!views().size())
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

    foreach (auto *item, selectedItems()) {
        if (auto node = dynamic_cast<QDMGraphicsNode *>(item)) {
            nodes.push_back(node);
        } else if (auto link = dynamic_cast<QDMGraphicsLinkFull *>(item)) {
            links.push_back(link);
        }
    }

    for (auto *link: links) {
        removeLink(link);
    }
    for (auto *node: nodes) {
        removeNode(node);
    }
}

/*std::vector<QDMGraphicsNode *> QDMGraphicsScene::getVisibleNodes() const
{
    std::vector<QDMGraphicsNode *> res;
    for (auto *node: nodes) {
        res.push_back(node);
    }
    return res;
}*/

std::vector<QDMGraphicsScene *> QDMGraphicsScene::getChildScenes() const
{
    std::vector<QDMGraphicsScene *> res;
    for (auto *node: nodes) {
        if (auto subnet = node->getSubnetScene())
            res.push_back(subnet);
    }
    return res;
}

void QDMGraphicsScene::addNode(QDMGraphicsNode *node) {
    addItem(node);
    nodes.emplace(node);
}

ZENO_NAMESPACE_END
