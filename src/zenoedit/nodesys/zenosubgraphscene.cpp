#include "zenosubgraphscene.h"
#include <zenoui/model/subgraphmodel.h>
#include "zenonode.h"
#include "subnetnode.h"
#include "zenolink.h"
#include <zenoui/model/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/nodesys/nodegrid.h>
#include <zenoui/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"


ZenoSubGraphScene::ZenoSubGraphScene(QObject *parent)
    : QGraphicsScene(parent)
    , m_tempLink(nullptr)
{
    ZtfUtil &inst = ZtfUtil::GetInstance();
    m_nodeParams = inst.toUtilParam(inst.loadZtf(":/templates/node-example.xml"));
    // bsp tree index causes crash when removeItem and delete item. for safety, disable it.
    // https://stackoverflow.com/questions/38458830/crash-after-qgraphicssceneremoveitem-with-custom-item-class
    setItemIndexMethod(QGraphicsScene::NoIndex);
}

ZenoSubGraphScene::~ZenoSubGraphScene()
{
}

void ZenoSubGraphScene::onViewTransformChanged(qreal factor)
{
}

void ZenoSubGraphScene::initModel(const QModelIndex& index)
{
    m_subgIdx = index;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    Q_ASSERT(pGraphsModel);

    disconnect(pGraphsModel, SIGNAL(reloaded(const QModelIndex&)), this, SLOT(reload(const QModelIndex&)));
    disconnect(pGraphsModel, SIGNAL(clearLayout(const QModelIndex&)), this, SLOT(clearLayout(const QModelIndex&)));
    disconnect(pGraphsModel, SIGNAL(_dataChanged(const QModelIndex&, const QModelIndex&, int)), this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, int)));
    disconnect(pGraphsModel, SIGNAL(_rowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)));
	disconnect(pGraphsModel, SIGNAL(_rowsInserted(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsInserted(const QModelIndex&, const QModelIndex&, int, int)));

    for (int r = 0; r < pGraphsModel->itemCount(m_subgIdx); r++)
    {
        const QModelIndex& idx = pGraphsModel->index(r, m_subgIdx);
        ZenoNode* pNode = createNode(idx, m_nodeParams);
        pNode->initUI(m_subgIdx, idx);
        addItem(pNode);
        const QString& nodeid = pNode->nodeId();
        m_nodes[nodeid] = pNode;
        connect(pNode, SIGNAL(socketPosInited(const QString&, const QString&, bool)),
                this, SLOT(onSocketPosInited(const QString&, const QString&, bool)));
    }

    for (auto it : m_nodes)
    {
        ZenoNode *node = it.second;
        const QString& id = node->nodeId();
        const INPUT_SOCKETS& inputs = node->inputParams();
        for (QString inSock : inputs.keys())
        {
            const INPUT_SOCKET& inputSocket = inputs[inSock];
            for (QString outId : inputSocket.outNodes.keys())
            {
                for (QString outSock : inputSocket.outNodes[outId].keys())
                {
                    ZenoNode* outNode = m_nodes[outId];
                    const QPointF &outSockPos = outNode->getPortPos(false, outSock);
                    EdgeInfo info(outId, id, outSock, inSock);
                    ZenoFullLink *pEdge = new ZenoFullLink(info);
                    pEdge->updatePos(outSockPos, node->getPortPos(true, inSock));
                    addItem(pEdge);
                    m_links.insert(std::make_pair(info, pEdge));
                    outNode->toggleSocket(false, outSock, true);
                }
            }
            if (!inputSocket.outNodes.isEmpty())
                node->toggleSocket(true, inSock, true);
        }
    }

    //a more effecient way is collect scene togther and send msg to specific scene.
	connect(pGraphsModel, SIGNAL(reloaded(const QModelIndex&)), this, SLOT(reload(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(clearLayout(const QModelIndex&)), this, SLOT(clearLayout(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(_dataChanged(const QModelIndex&, const QModelIndex&, int)), this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, int)));
    connect(pGraphsModel, SIGNAL(_rowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)));
    connect(pGraphsModel, SIGNAL(_rowsInserted(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsInserted(const QModelIndex&, const QModelIndex&, int, int)));
}

ZenoNode* ZenoSubGraphScene::createNode(const QModelIndex& idx, const NodeUtilParam& params)
{
    NODE_TYPE type = static_cast<NODE_TYPE>(idx.data(ROLE_NODETYPE).toInt());
    switch (type)
    {
        case SUBINPUT_NODE: return new SubInputNode(params);
        default:
            return new ZenoNode(params);
    }
}

void ZenoSubGraphScene::onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
    if (subGpIdx != m_subgIdx)
        return;

	QString id = idx.data(ROLE_OBJID).toString();

	if (role == ROLE_OBJPOS)
	{
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
	    QPointF pos = idx.data(ROLE_OBJPOS).toPointF();
        m_nodes[id]->setPos(pos);
		updateLinkPos(m_nodes[id], pos);
	}
	if (role == ROLE_INPUTS || role == ROLE_OUTPUTS)
	{
				//it's diffcult to detect which link has changed on this method.
                //use ROLE_ADDLINK/ROLE_REMOVELINK instead.
	}
    if (role == ROLE_ADDLINK)
    {
        EdgeInfo info = idx.data(role).value<EdgeInfo>();
        if (info.inputNode.isEmpty() || info.inputSock.isEmpty() || info.outputNode.isEmpty() || info.outputSock.isEmpty())
        {
            return;
        }

	    Q_ASSERT(m_links.find(info) == m_links.end());
		if (m_links.find(info) == m_links.end())
		{
			onLinkChanged(true, info.outputNode, info.outputSock, info.inputNode, info.inputSock);
		}
    }
    if (role == ROLE_REMOVELINK)
    {
        EdgeInfo info = idx.data(ROLE_REMOVELINK).value<EdgeInfo>();
		if (info.inputNode.isEmpty() || info.inputSock.isEmpty() || info.outputNode.isEmpty() || info.outputSock.isEmpty())
		{
            return;
		}

        Q_ASSERT(m_links.find(info) != m_links.end());
        if (m_links.find(info) != m_links.end())
        {
            onLinkChanged(false, info.outputNode, info.outputSock, info.inputNode, info.inputSock);
        }
    }
	if (role == ROLE_OPTIONS)
	{
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
		int options = idx.data(ROLE_OPTIONS).toInt();
		m_nodes[id]->onOptionsUpdated(options);
	}
	if (role == ROLE_COLLASPED)
	{
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
		bool bCollasped = idx.data(ROLE_COLLASPED).toBool();
		m_nodes[id]->onCollaspeUpdated(bCollasped);
	}
    if (role == ROLE_MODIFY_PARAM)
    {
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        PARAM_INFO info = idx.data(ROLE_MODIFY_PARAM).value<PARAM_INFO>();
        m_nodes[id]->onParamUpdated(info.name, info.value);
    }
    if (role == ROLE_OBJNAME)
    {
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        m_nodes[id]->onNameUpdated(idx.data(ROLE_OBJNAME).toString());
    }
}

QRectF ZenoSubGraphScene::nodesBoundingRect() const
{
    QRectF boundingRect;
    for (auto item : m_nodes)
    {
        boundingRect |= item.second->sceneBoundingRect();
    }
    return boundingRect;
}

QModelIndex ZenoSubGraphScene::subGraphIndex() const
{
    return m_subgIdx;
}

void ZenoSubGraphScene::select(const QString& id)
{
    clearSelection();
    Q_ASSERT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id]->setSelected(true);
}

void ZenoSubGraphScene::undo()
{
    //todo
    //m_subgraphModel->undo();
}

void ZenoSubGraphScene::redo()
{
    //todo
}

void ZenoSubGraphScene::copy()
{
    QList<QGraphicsItem*> selItems = this->selectedItems();
    if (selItems.isEmpty())
        return;

    //todo: write json format data to clipboard.

    QMap<EdgeInfo, ZenoFullLink *> selLinks;
    QMap<QString, ZenoNode*> selNodes;

    for (auto item : selItems)
    {
        if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
        {
            selNodes.insert(pNode->nodeId(), pNode);
        }
    }
    for (auto item : selItems)
    {
        if (ZenoFullLink* pLink = qgraphicsitem_cast<ZenoFullLink*>(item))
        {
            const EdgeInfo& info = pLink->linkInfo();
            if (selNodes.find(info.inputNode) == selNodes.end() ||
                selNodes.find(info.outputNode) == selNodes.end())
            {
                continue;
            }
            selLinks[info] = pLink;
        }
    }
    if (selNodes.isEmpty())
    {
        QApplication::clipboard()->clear();
    }

	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	Q_ASSERT(pGraphsModel);

    QMap<QString, QString> oldToNew;
    QMap<QString, NODE_DATA> newNodes;
    QList<NODE_DATA> vecNodes;
    for (auto pNode : selNodes)
    {
        QString currNode = pNode->nodeId();
        NODE_DATA data = pGraphsModel->itemData(pNode->index(), m_subgIdx);
        QString oldId = data[ROLE_OBJID].toString();
        const QString &newId = UiHelper::generateUuid(data[ROLE_OBJNAME].toString());
        data[ROLE_OBJID] = newId;
        oldToNew[oldId] = newId;

        //clear any connections.
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        INPUT_SOCKETS newInputs;
        for (auto inSock : inputs.keys())
        {
            INPUT_SOCKET& socket = inputs[inSock];
            socket.outNodes.clear();
        }
        data[ROLE_INPUTS] = QVariant::fromValue(inputs);

        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (auto outSock : outputs.keys())
        {
            OUTPUT_SOCKET& socket = outputs[outSock];
            socket.inNodes.clear();
        }
        data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

        newNodes[newId] = data;
    }

    for (auto edge : selLinks.keys())
    {
        const QString &outOldId = edge.outputNode;
        const QString &inOldId = edge.inputNode;

        const QString &outId = oldToNew[outOldId];
        const QString &inId = oldToNew[inOldId];

        const QString& outSock = edge.outputSock;
        const QString& inSock = edge.inputSock;

        //out link
        NODE_DATA& outData = newNodes[outId];
        OUTPUT_SOCKETS outputs = outData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        SOCKET_INFO &newOutSocket = outputs[outSock].inNodes[inId][inSock];

        QModelIndex tempIdx = pGraphsModel->index(outOldId, m_subgIdx);
        
        NODE_DATA outOldData = pGraphsModel->itemData(tempIdx, m_subgIdx);
        OUTPUT_SOCKETS oldOutputs = outOldData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        const SOCKET_INFO &oldOutSocket = oldOutputs[outSock].inNodes[inOldId][inSock];
        newOutSocket = oldOutSocket;
        newOutSocket.nodeid = inId;
        outData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

        //in link
        NODE_DATA& inData = newNodes[inId];
        INPUT_SOCKETS inputs = inData[ROLE_INPUTS].value<INPUT_SOCKETS>();
        SOCKET_INFO &newInSocket = inputs[inSock].outNodes[outId][outSock];
        
        tempIdx = pGraphsModel->index(inOldId, m_subgIdx);
        NODE_DATA inOldData = pGraphsModel->itemData(tempIdx, m_subgIdx);
        INPUT_SOCKETS oldInputs = inOldData[ROLE_INPUTS].value<INPUT_SOCKETS>();
        const SOCKET_INFO &oldInSocket = oldInputs[inSock].outNodes[outOldId][outSock];
        newInSocket = oldInSocket;
        newInSocket.nodeid = outId;
        inData[ROLE_INPUTS] = QVariant::fromValue(inputs);
    }

    NODES_MIME_DATA* pNodesData = new NODES_MIME_DATA;
    for (auto node : newNodes)
    {
        INPUT_SOCKETS inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
        OUTPUT_SOCKETS outputs = node[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        pNodesData->m_vecNodes.push_back(node);
    }

    QMimeData *pMimeData = new QMimeData;
    pMimeData->setUserData(MINETYPE_MULTI_NODES, pNodesData);
    QApplication::clipboard()->setMimeData(pMimeData);
}

void ZenoSubGraphScene::paste(QPointF pos)
{
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

    if (QObjectUserData *pUserData = pMimeData->userData(MINETYPE_MULTI_NODES))
    {
        NODES_MIME_DATA* pNodesData = static_cast<NODES_MIME_DATA*>(pUserData);
        if (pNodesData->m_vecNodes.isEmpty())
            return;

        QPointF offset = pos - pNodesData->m_vecNodes[0][ROLE_OBJPOS].toPointF();

        //todo: pGraphsModel->beginMacro("paste nodes");
        QList<NODE_DATA> datas;
        for (int i = 0; i < pNodesData->m_vecNodes.size(); i++)
        {
            NODE_DATA& data = pNodesData->m_vecNodes[i];
            QPointF orginalPos = data[ROLE_OBJPOS].toPointF();
            data[ROLE_OBJPOS] = orginalPos + offset;
        }
        pGraphsModel->appendNodes(pNodesData->m_vecNodes, m_subgIdx);

        clearSelection();
        for (auto node : pNodesData->m_vecNodes)
        {
            const QString &id = node[ROLE_OBJID].toString();
            Q_ASSERT(m_nodes.find(id) != m_nodes.end());
            m_nodes[id]->setSelected(true);
        }

        //pGraphsModel->endMacro();
    }
}

QPointF ZenoSubGraphScene::getSocketPos(bool bInput, const QString &nodeid, const QString &portName)
{
    auto it = m_nodes.find(nodeid);
    Q_ASSERT(it != m_nodes.end());
    QPointF pos = it->second->getPortPos(bInput, portName);
    return pos;
}

void ZenoSubGraphScene::reload(const QModelIndex& subGpIdx)
{
    if (subGpIdx != m_subgIdx)
    {
        clear();
        initModel(subGpIdx);
    }
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

        EdgeInfo info(outId, inId, outPort, inPort);
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        pGraphsModel->addLink(info, m_subgIdx);

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

void ZenoSubGraphScene::clearLayout(const QModelIndex& subGpIdx)
{
    if (subGpIdx != m_subgIdx)
    {
		m_nodes.clear();
		m_links.clear();
		clear();
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

void ZenoSubGraphScene::onRowsAboutToBeRemoved(const QModelIndex& subgIdx, const QModelIndex &parent, int first, int last)
{
    if (subgIdx != m_subgIdx)
        return;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    for (int r = first; r <= last; r++)
    {
        QModelIndex idx = pGraphsModel->index(r, m_subgIdx);
        QString id = idx.data(ROLE_OBJID).toString();
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        ZenoNode* pNode = m_nodes[id];
        removeItem(pNode);
        delete pNode;
        m_nodes.erase(id);
    }
}

void ZenoSubGraphScene::onRowsInserted(const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last)
{
    if (subgIdx != m_subgIdx)
        return;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex idx = pGraphsModel->index(first, m_subgIdx);
    ZenoNode *pNode = new ZenoNode(m_nodeParams);
    pNode->initUI(m_subgIdx, idx);
    QString id = pNode->nodeId();
    addItem(pNode);
    m_nodes[id] = pNode;
	connect(pNode, SIGNAL(socketPosInited(const QString&, const QString&, bool)),
		this, SLOT(onSocketPosInited(const QString&, const QString&, bool)));
}

void ZenoSubGraphScene::onSocketPosInited(const QString& nodeid, const QString& sockName, bool bInput)
{
    Q_ASSERT(m_nodes.find(nodeid) != m_nodes.end());
    if (bInput)
    {
        ZenoNode* pInputNode = m_nodes[nodeid];
        QPointF pos = pInputNode->getPortPos(true, sockName);
        for (auto itLink : m_links)
        {
            const EdgeInfo& info = itLink.first;
            if (info.inputNode == nodeid && info.inputSock == sockName)
            {
                itLink.second->initDstPos(pos);
            }
        }
    }
    else
    {
        ZenoNode* pOutputNode = m_nodes[nodeid];
        QPointF pos = pOutputNode->getPortPos(false, sockName);
        for (auto itLink : m_links)
        {
            const EdgeInfo &info = itLink.first;
            if (info.outputNode == nodeid && info.outputSock == sockName)
			{
				itLink.second->initSrcPos(pos);
			}
		}
    }
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
        QList<ZenoNode*> nodes;
        QList<ZenoFullLink*> links;
        for (auto item : selItems)
        {
            if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item)) {
                nodes.append(pNode);
            } else if (ZenoFullLink *pLink = qgraphicsitem_cast<ZenoFullLink *>(item)) {
                links.append(pLink);
            }
        }
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        //todo:
        //pGraphsModel->beginMacro("remove nodes and links");
        for (auto item : links)
        {
            const EdgeInfo &info = item->linkInfo();
            pGraphsModel->removeLink(info, m_subgIdx);
        }
        for (auto item : nodes)
        {
            const QPersistentModelIndex &index = item->index();
            pGraphsModel->removeNode(index.row(), m_subgIdx);
        }
        //pGraphsModel->endMacro();
    }
    QGraphicsScene::keyPressEvent(event);
}