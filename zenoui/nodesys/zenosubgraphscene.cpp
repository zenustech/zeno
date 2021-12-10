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

                ZenoLinkFull *pEdge = new ZenoLinkFull(this, fromId, outputPort, id, portName);
                addItem(pEdge);
            }
        }
    }

    connect(m_subgraphModel, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)),
        this, SLOT(void onDataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&)));
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
}

void ZenoSubGraphScene::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        for (auto item : this->selectedItems())
        {
            ZenoNode *pNode = qgraphicsitem_cast<ZenoNode*>(item);
            const QPersistentModelIndex &index = pNode->index();
            m_subgraphModel->removeRows(index.row(), 1);
            removeItem(pNode);
            delete pNode;
        }
    }
    QGraphicsScene::keyPressEvent(event);
}