#include "zenosubgraphscene.h"
#include "../model/subgraphmodel.h"
#include "zenonode.h"
#include "zenolink.h"
#include "../model/modelrole.h"


ZenoSubGraphScene::ZenoSubGraphScene(QObject *parent)
    : QGraphicsScene(parent)
    , m_subgraphModel(nullptr)
    , m_tempLink(nullptr)
{
    ZtfUtil &inst = ZtfUtil::GetInstance();
    m_nodeParams = inst.toUtilParam(inst.loadZtf(":/templates/node-example.xml"));
    // bsp tree index causes crash when removeItem and delete item. for safety, disable it.
    // https://stackoverflow.com/questions/38458830/crash-after-qgraphicssceneremoveitem-with-custom-item-class
    setItemIndexMethod(QGraphicsScene::NoIndex);
}

void ZenoSubGraphScene::initModel(SubGraphModel* pModel)
{
    m_subgraphModel = pModel;
    int n = m_subgraphModel->rowCount();
    for (int r = 0; r < n; r++)
    {
        const QModelIndex& idx = m_subgraphModel->index(r, 0);
        ZenoNode* pNode = new ZenoNode(m_nodeParams);
        pNode->init(idx);
        addItem(pNode);
        m_nodes.insert(std::make_pair(pNode->nodeId(), pNode));
    }

    for (auto it : m_nodes)
    {
        ZenoNode *node = it.second;
        const QString& id = node->nodeId();
        const INPUT_SOCKETS& inputs = node->inputParams();
        for (QString inputSock : inputs.keys())
        {
            const INPUT_SOCKET& inSock = inputs[inputSock];
            for (QString outId : inSock.outNodes.keys())
            {
                for (QString outSock : inSock.outNodes[outId].keys())
                {
                    ZenoNode* outNode = m_nodes[outId];
                    const QPointF &outSockPos = outNode->getPortPos(false, outSock);
                    EdgeInfo info(outId, id, outSock, inputSock);
                    ZenoFullLink *pEdge = new ZenoFullLink(info);
                    pEdge->updatePos(outSockPos, node->getPortPos(true, inputSock));
                    addItem(pEdge);
                    m_links.insert(std::make_pair(info, pEdge));
                    outNode->toggleSocket(false, outSock, true);
                }
            }
            if (!inSock.outNodes.isEmpty())
                node->toggleSocket(true, inputSock, true);
        }
    }

    connect(m_subgraphModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)));
    connect(m_subgraphModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
    connect(m_subgraphModel, SIGNAL(rowsInserted(const QModelIndex &, int , int)),
        this, SLOT(onRowsInserted(const QModelIndex&, int, int)));
    connect(m_subgraphModel, SIGNAL(linkChanged(bool, const QString&, const QString&, const QString&, const QString&)),
        this, SLOT(onLinkChanged(bool, const QString &, const QString &, const QString &, const QString &)));
}

QPointF ZenoSubGraphScene::getSocketPos(bool bInput, const QString &nodeid, const QString &portName)
{
    auto it = m_nodes.find(nodeid);
    Q_ASSERT(it != m_nodes.end());
    QPointF pos = it->second->getPortPos(bInput, portName);
    return pos;
}

bool ZenoSubGraphScene::_enableLink(const QString &outputNode, const QString &outputSocket, const QString& inputNode, const QString& inputSocket)
{
    return false;
}

void ZenoSubGraphScene::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QList<QGraphicsItem *> items = this->items(event->scenePos());
    ZenoSocketItem* pSocket = nullptr;
    for (auto item : items)
    {
        if (pSocket = qgraphicsitem_cast<ZenoSocketItem*>(item))
            break;
    }

    if (!m_tempLink && pSocket)
    {
        SOCKET_INFO info = pSocket->getSocketInfo();
        QPointF wtf = pSocket->scenePos();
        m_tempLink = new ZenoTempLink(info);
        addItem(m_tempLink);
        pSocket->toggle(true);
        return;
    }
    else if (m_tempLink && pSocket)
    {
        SOCKET_INFO info1, info2;
        m_tempLink->getFixedInfo(info1);
        info2 = pSocket->getSocketInfo();
        if (info1.binsock == info2.binsock)
            return;

        QString outId, outPort, inId, inPort;
        QPointF outPos, inPos;
        if (info1.binsock) {
            outId = info2.nodeid;
            outPort = info2.name;
            outPos = info2.pos;
            inId = info1.nodeid;
            inPort = info1.name;
            inPos = info1.pos;
        } else {
            outId = info1.nodeid;
            outPort = info1.name;
            outPos = info1.pos;
            inId = info2.nodeid;
            inPort = info2.name;
            inPos = info2.pos;
        }
        m_subgraphModel->addLink(outId, outPort, inId, inPort);

        removeItem(m_tempLink);
        delete m_tempLink;
        m_tempLink = nullptr;
        return;
    }
    else if (m_tempLink)
    {
        SOCKET_INFO info;
        m_tempLink->getFixedInfo(info);
        m_nodes[info.nodeid]->toggleSocket(info.binsock, info.name, false);

        removeItem(m_tempLink);
        delete m_tempLink;
        m_tempLink = nullptr;
        return;
    }

    QGraphicsScene::mousePressEvent(event);
}

void ZenoSubGraphScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_tempLink) {
        QPointF pos = event->scenePos();
        m_tempLink->setFloatingPos(pos);
    }
    QGraphicsScene::mouseMoveEvent(event);
}

void ZenoSubGraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsScene::mouseReleaseEvent(event);
}

void ZenoSubGraphScene::onNewNodeCreated()
{

}

void ZenoSubGraphScene::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    //model to scene.
    for (int r = topLeft.row(); r <= bottomRight.row(); r++)
    {
        QModelIndex idx = m_subgraphModel->index(r, 0);
        QString id = idx.data(ROLE_OBJID).toString();
        for (int role : roles)
        {
            if (role == ROLE_OBJPOS)
            {
                QPointF pos = idx.data(ROLE_OBJPOS).toPointF();
                updateLinkPos(m_nodes[id], pos);
            }
            if (role == ROLE_INPUTS || role == ROLE_OUTPUTS)
            {
                //it's diffcult to detect which link has changed.
            }
        }
    }
}

void ZenoSubGraphScene::onLinkChanged(bool bAdd, const QString& outputId, const QString& outputSock, const QString& inputId, const QString& inputSock)
{
    ZenoNode *pInputNode = m_nodes[inputId];
    ZenoNode *pOutputNode = m_nodes[outputId];
    if (bAdd)
    {
        EdgeInfo info(outputId, inputId, outputSock, inputSock);
        ZenoFullLink *pEdge = new ZenoFullLink(info);
        const QPointF &outPos = pOutputNode->getPortPos(false, outputSock);
        const QPointF &inPos = pInputNode->getPortPos(true, inputSock);
        pEdge->updatePos(outPos, inPos);
        addItem(pEdge);

        m_links.insert(std::make_pair(info, pEdge));
        pInputNode->toggleSocket(true, inputSock, true);
        pOutputNode->toggleSocket(false, outputSock, true);
    }
    else
    {
        EdgeInfo info(outputId, inputId, outputSock, inputSock);
        ZenoFullLink *pLink = m_links[info];
        removeItem(pLink);
        delete pLink;

        m_links.erase(info);

        if (pInputNode)
        {
            auto const &inSocks = pInputNode->inputParams();
            const auto inSock = inSocks[inputSock];
            if (inSock.outNodes.isEmpty()) {
                pInputNode->toggleSocket(true, inputSock, false);
            }
        }
        if (pOutputNode)
        {
            auto const &outSocks = pOutputNode->outputParams();
            if (outSocks.find(outputSock).value().inNodes.isEmpty()) {
                pOutputNode->toggleSocket(false, outputSock, false);
            }
        }
    }
}

void ZenoSubGraphScene::onRowsAboutToBeRemoved(const QModelIndex &parent, int first, int last)
{
    for (int r = first; r <= last; r++)
    {
        QModelIndex idx = m_subgraphModel->index(r, 0);
        QString id = idx.data(ROLE_OBJID).toString();
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        ZenoNode* pNode = m_nodes[id];
        removeItem(pNode);
        delete pNode;
        m_nodes.erase(id);
    }
}

void ZenoSubGraphScene::onRowsInserted(const QModelIndex& parent, int first, int last)
{
    QModelIndex idx = m_subgraphModel->index(first, 0);
    ZenoNode *pNode = new ZenoNode(m_nodeParams);
    pNode->init(idx);
    QString id = pNode->nodeId();
    addItem(pNode);
    m_nodes[id] = pNode;
}

void ZenoSubGraphScene::updateLinkPos(ZenoNode* pNode, QPointF newPos)
{
    if (!pNode)
        return;
    pNode->setPos(newPos);
    const QString& currNode = pNode->nodeId();
    const INPUT_SOCKETS& inputs = pNode->inputParams();
    for (QString inSock : inputs.keys())
    {
        const auto& outNodes = inputs[inSock].outNodes;
        for (QString outNode : outNodes.keys())
        {
            for (QString outSock : outNodes[outNode].keys())
            {
                const SOCKET_INFO& socketinfo = outNodes[outNode][outSock];
                const QPointF &outputPos = m_nodes[outNode]->getPortPos(false, outSock);
                const QPointF &inputPos = pNode->getPortPos(true, inSock);

                EdgeInfo info(outNode, currNode, outSock, inSock);
                ZenoFullLink *pLink = m_links[info];
                Q_ASSERT(pLink);
                pLink->updatePos(outputPos, inputPos);
            }
        }
    }

    const OUTPUT_SOCKETS &outputs = pNode->outputParams();
    for (QString outputPort : outputs.keys())
    {
        const QPointF &outputPos = pNode->getPortPos(false, outputPort);
        for (QString inNode : outputs[outputPort].inNodes.keys())
        {
            const SOCKETS_INFO& sockets = outputs[outputPort].inNodes[inNode];
            for (QString inSock : sockets.keys())
            {
                QPointF sockPos = m_nodes[inNode]->getPortPos(true, inSock);

                EdgeInfo info(currNode, inNode, outputPort, inSock);
                ZenoFullLink *pLink = m_links[info];
                pLink->updatePos(outputPos, sockPos);
            }
        }
    }
}

void ZenoSubGraphScene::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        QList<QGraphicsItem*> selItems = this->selectedItems();
        QList<ZenoNode *> nodes;
        QList<ZenoFullLink *> links;
        for (auto item : selItems)
        {
            if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item)) {
                nodes.append(pNode);
            } else if (ZenoFullLink *pLink = qgraphicsitem_cast<ZenoFullLink *>(item)) {
                links.append(pLink);
            }
        }
        for (auto item : links) {
            const EdgeInfo &info = item->linkInfo();
            m_subgraphModel->removeLink(info.srcNode, info.srcPort, info.dstNode, info.dstPort);
        }
        for (auto item : nodes)
        {
            const QPersistentModelIndex &index = item->index();
            m_subgraphModel->removeNode(index.row());
        }
    }
    QGraphicsScene::keyPressEvent(event);
}