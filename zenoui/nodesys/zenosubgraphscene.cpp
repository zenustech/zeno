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
        pNode->show();
        QPointF pos = idx.data(ROLE_OBJPOS).toPointF();
        const QString& id = idx.data(ROLE_OBJID).toString();
        pNode->setPos(pos);
        addItem(pNode);
        pNode->updateSocketPos();
        m_nodes.insert(std::make_pair(id, pNode));
    }

    for (auto it : m_nodes)
    {
        ZenoNode *node = it.second;
        const QString& id = node->nodeId();
        const QJsonObject& inputs = node->inputParams();
        for (QString inputPort : inputs.keys())
        {
            const QJsonValue& fromSocket = inputs.value(inputPort);
            Q_ASSERT(fromSocket.isArray());
            const QJsonArray& arr = fromSocket.toArray();
            Q_ASSERT(arr.size() == 3);
            if (!arr[0].isNull())
            {
                const QString &fromId = arr[0].toString();
                const QString &outputPort = arr[1].toString();
                ZenoNode* fromNode = m_nodes[fromId];
                const QPointF& fromPos = fromNode->getPortPos(false, outputPort);

                EdgeInfo info(fromId, id, outputPort, inputPort);
                ZenoFullLink *pEdge = new ZenoFullLink(info);
                pEdge->updatePos(fromPos, node->getPortPos(true, inputPort));
                addItem(pEdge);
                m_links.insert(std::make_pair(info, pEdge));
            }
        }
    }

    connect(m_subgraphModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)));
    connect(m_subgraphModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex &, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
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

    //
    if (!m_tempLink && pSocket)
    {
        SOCKET_INFO info = pSocket->getSocketInfo();
        QPointF wtf = pSocket->scenePos();
        m_tempLink = new ZenoTempLink(info);
        addItem(m_tempLink);
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

        EdgeInfo info(outId, inId, outPort, inPort);
        ZenoFullLink *pEdge = new ZenoFullLink(info);
        pEdge->updatePos(outPos, inPos);
        addItem(pEdge);
        m_links.insert(std::make_pair(info, pEdge));

        //todo: update model.
        //actually should update model, and create new link by model!

        removeItem(m_tempLink);
        delete m_tempLink;
        m_tempLink = nullptr;
        return;
    }
    else if (m_tempLink)
    {
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
                updateNodePos(m_nodes[id], pos);
            }
            if (role == ROLE_INPUTS || role == ROLE_OUTPUTS)
            {
                //it's diffcult to detect which link has changed.
            }
        }
    }
}



void ZenoSubGraphScene::onLinkChanged(bool bAdd, const QString& outputId, const QString& outputPort, const QString& inputId, const QString& inputPort)
{
    if (bAdd)
    {
        //todo
    }
    else
    {
        EdgeInfo info(outputId, inputId, outputPort, inputPort);
        ZenoFullLink *pLink = m_links[info];
        removeItem(pLink);
        delete pLink;
        m_links.erase(info);
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

void ZenoSubGraphScene::updateNodePos(ZenoNode* pNode, QPointF newPos)
{
    pNode->setPos(newPos);
    const QString& id = pNode->nodeId();
    const QJsonObject& inputs = pNode->inputParams();
    const QJsonObject& outputs = pNode->outputParams();
    for (QString inputPort : inputs.keys())
    {
        const QJsonArray &arr = inputs.value(inputPort).toArray();
        if (arr[0].isNull()) continue;
        const QString &outputId = arr[0].toString();
        const QString &outputPort = arr[1].toString();
        const QPointF& outputPos = m_nodes[outputId]->getPortPos(false, outputPort);
        const QPointF& inputPos = pNode->getPortPos(true, inputPort);

        EdgeInfo info(outputId, id, outputPort, inputPort);
        ZenoFullLink* pLink = m_links[info];
        Q_ASSERT(pLink);
        pLink->updatePos(outputPos, inputPos);
    }

    /* output format :
       {
           "port1" : {
                "node1": "port_in_node1",
                "node2": "port_in_node2",
           },
           "port2" : {
                    ...
           }
       }
    */
    for (QString outputPort : outputs.keys())
    {
        const QPointF &outputPos = pNode->getPortPos(false, outputPort);
        const QJsonObject &inputObj = outputs.value(outputPort).toObject();
        if (inputObj.isEmpty()) continue;

        for (auto inputId : inputObj.keys())
        {
            const QString &inputPort = inputObj.value(inputId).toString();
            const QPointF &inputPos = m_nodes[inputId]->getPortPos(true, inputPort);

            EdgeInfo info(id, inputId, outputPort, inputPort);
            ZenoFullLink* pLink = m_links[info];
            pLink->updatePos(outputPos, inputPos);
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
        for (auto item : nodes) {
            const QPersistentModelIndex &index = item->index();
            m_subgraphModel->removeNode(index);
        }
    }
    QGraphicsScene::keyPressEvent(event);
}