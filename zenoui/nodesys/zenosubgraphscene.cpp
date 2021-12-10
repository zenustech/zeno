#include "zenosubgraphscene.h"
#include "../model/subgraphmodel.h"
#include "zenonode.h"
#include "zenolink.h"
#include "../model/modelrole.h"


ZenoSubGraphScene::ZenoSubGraphScene(QObject *parent)
    : QGraphicsScene(parent)
    , m_subgraphModel(nullptr)
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
        m_nodes.insert(std::make_pair(id, pNode));
    }

    for (auto it : m_nodes)
    {
        ZenoNode *node = it.second;
        const QString& id = node->nodeId();
        const QJsonObject& inputs = node->inputParams();
        for (QString portName : inputs.keys())
        {
            const QJsonValue& fromSocket = inputs.value(portName);
            Q_ASSERT(fromSocket.isArray());
            const QJsonArray& arr = fromSocket.toArray();
            Q_ASSERT(arr.size() == 3);
            if (!arr[0].isNull())
            {
                const QString &fromId = arr[0].toString();
                const QString &outputPort = arr[1].toString();
                ZenoNode* fromNode = m_nodes[fromId];
                const QPointF& fromPos = fromNode->getPortPos(false, outputPort);

                EdgeInfo info(fromId, id, outputPort, portName);
                ZenoLinkFull *pEdge = new ZenoLinkFull;
                pEdge->updatePos(fromPos, node->getPortPos(true, portName));
                addItem(pEdge);
                m_links.insert(std::make_pair(info, pEdge));
            }
        }
    }

    connect(m_subgraphModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)));
    connect(m_subgraphModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex &, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
}

QPointF ZenoSubGraphScene::getSocketPos(bool bInput, const QString &nodeid, const QString &portName)
{
    auto it = m_nodes.find(nodeid);
    Q_ASSERT(it != m_nodes.end());
    QPointF pos = it->second->getPortPos(bInput, portName);
    return pos;
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
        }
    }
}

void ZenoSubGraphScene::onRowsAboutToBeRemoved(const QModelIndex &parent, int first, int last)
{
    for (int r = first; r <= last; r++)
    {
        QModelIndex idx = m_subgraphModel->index(r, 0);
        QString id = idx.data(ROLE_OBJID).toString();
        ZenoNode* pNode = m_nodes[id];

        //remove edge
        const QJsonObject &inputs = pNode->inputParams();
        const QJsonObject &outputs = pNode->outputParams();

        //todo: get these by iter
        for (QString inputPort : inputs.keys())
        {
            const QJsonArray &arr = inputs.value(inputPort).toArray();
            if (arr[0].isNull()) continue;
            const QString &outputId = arr[0].toString();
            const QString &outputPort = arr[1].toString();

            //should remove params from other node and update these node!
            
            /*
            EdgeInfo info(outputId, id, outputPort, inputPort);
            ZenoLinkFull *pLink = m_links[info];
            removeItem(pLink);
            delete pLink;
            m_links.erase(info);
            */
        }
        for (QString outputPort : outputs.keys())
        {
            const QJsonArray &arr = outputs.value(outputPort).toArray();
            if (arr[0].isNull()) continue;
            const QString &inputId = arr[0].toString();
            const QString &inputPort = arr[1].toString();

            /*
            EdgeInfo info(id, inputId, outputPort, inputPort);
            ZenoLinkFull *pLink = m_links[info];
            removeItem(pLink);
            delete pLink;
            m_links.erase(info);
            */
        }

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
        ZenoLinkFull* pLink = m_links[info];
        pLink->updatePos(outputPos, inputPos);
    }
    for (QString outputPort : outputs.keys())
    {
        const QJsonArray &arr = outputs.value(outputPort).toArray();
        if (arr[0].isNull()) continue;
        const QString &inputId = arr[0].toString();
        const QString &inputPort = arr[1].toString();
        const QPointF &inputPos = m_nodes[inputId]->getPortPos(true, inputPort);
        const QPointF &outputPos = pNode->getPortPos(false, outputPort);

        EdgeInfo info(id, inputId, outputPort, inputPort);
        ZenoLinkFull *pLink = m_links[info];
        pLink->updatePos(outputPos, inputPos);
    }
}

void ZenoSubGraphScene::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        for (auto item : this->selectedItems())
        {
            if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
            {
                const QPersistentModelIndex &index = pNode->index();
                //m_subgraphModel->removeRows(index.row(), 1);
                //m_subgraphModel->removeNode(index.row)
            }
            else if (ZenoLinkFull *pLink = qgraphicsitem_cast<ZenoLinkFull*>(item))
            {
                removeItem(pLink);
                delete pLink;
            }
        }
    }
    QGraphicsScene::keyPressEvent(event);
}