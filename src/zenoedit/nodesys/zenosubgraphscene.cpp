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
#include <zeno/utils/log.h>


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
    {//loaded nodes goes here
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
            QPointF inSockPos = node->getPortPos(true, inSock);
            const INPUT_SOCKET& inputSocket = inputs[inSock];
            for (const QPersistentModelIndex& linkIdx : inputSocket.linkIndice)
            {
                ZenoFullLink* pEdge = new ZenoFullLink(linkIdx);

                const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
                const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
                const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                ZenoNode* outNode = m_nodes[outId];
                const QPointF& outSockPos = outNode->getPortPos(false, outSock);

                pEdge->updatePos(outSockPos, inSockPos);
                addItem(pEdge);
                m_links[linkId] = pEdge;
                outNode->toggleSocket(false, outSock, true);
            }
        }
    }

    //a more effecient way is collect scene togther and send msg to specific scene.
	connect(pGraphsModel, SIGNAL(reloaded(const QModelIndex&)), this, SLOT(reload(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(clearLayout(const QModelIndex&)), this, SLOT(clearLayout(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(_dataChanged(const QModelIndex&, const QModelIndex&, int)), this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, int)));
    connect(pGraphsModel, SIGNAL(_rowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)));
    connect(pGraphsModel, SIGNAL(_rowsInserted(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsInserted(const QModelIndex&, const QModelIndex&, int, int)));

    //link sync:
    connect(pGraphsModel, &IGraphsModel::linkDataChanged, this, &ZenoSubGraphScene::onLinkDataChanged);
    connect(pGraphsModel, &IGraphsModel::linkAboutToBeInserted, this, &ZenoSubGraphScene::onLinkAboutToBeInserted);
    connect(pGraphsModel, &IGraphsModel::linkInserted, this, &ZenoSubGraphScene::onLinkInserted);
    connect(pGraphsModel, &IGraphsModel::linkAboutToBeRemoved, this, &ZenoSubGraphScene::onLinkAboutToBeRemoved);
    connect(pGraphsModel, &IGraphsModel::linkRemoved, this, &ZenoSubGraphScene::onLinkRemoved);

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
        //but link sync is managed by linkModel
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
    if (role == ROLE_MODIFY_SOCKET)
    {
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        QVariant var = idx.data(ROLE_MODIFY_SOCKET);
        if (var.isNull())
            return;
        SOCKET_UPDATE_INFO info = var.value<SOCKET_UPDATE_INFO>();
        m_nodes[id]->onSocketUpdated(info);
        const QString& oldName = info.oldinfo.name;
        const QString& newName = info.newInfo.name;
        
        //update links
        /*
        for (const EdgeInfo& edgeInfo : m_links.keys())
        {
            if (info.bInput)
            {
                if (edgeInfo.inputNode == id && edgeInfo.inputSock == oldName)
                {
                    EdgeInfo _edgeInfo(edgeInfo);
                    ZenoFullLink* pLink = m_links[edgeInfo];
                    _edgeInfo.inputSock = newName;
                    pLink->updateLink(_edgeInfo);
                    m_links[_edgeInfo] = pLink;
                    m_links.remove(edgeInfo);
                }
            }
            else
            {
                if (edgeInfo.outputNode == id && edgeInfo.outputSock == oldName)
                {
                    EdgeInfo _edgeInfo(edgeInfo);
                    ZenoFullLink* pLink = m_links[edgeInfo];
                    _edgeInfo.outputSock = newName;
                    pLink->updateLink(_edgeInfo);
                    m_links[_edgeInfo] = pLink;
                    m_links.remove(edgeInfo);
                }
            }
        }
        */
    }
    if (role == ROLE_OBJNAME)
    {
        Q_ASSERT(m_nodes.find(id) != m_nodes.end());
        m_nodes[id]->onNameUpdated(idx.data(ROLE_OBJNAME).toString());
    }
}

void ZenoSubGraphScene::onLinkDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
	if (subGpIdx != m_subgIdx)
		return;
}

void ZenoSubGraphScene::onLinkAboutToBeInserted(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last)
{
	if (subGpIdx != m_subgIdx)
		return;
}

void ZenoSubGraphScene::onLinkInserted(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last)
{
	if (subGpIdx != m_subgIdx)
		return;

    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex linkIdx = pGraphsModel->linkIndex(first);

	ZenoFullLink* pEdge = new ZenoFullLink(QPersistentModelIndex(linkIdx));

	const QString& linkId = linkIdx.data(ROLE_OBJID).toString();

    const QString& inId = linkIdx.data(ROLE_INNODE).toString();
    const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();
	const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
	const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

    ZenoNode* pNode = m_nodes[inId];
    const QPointF& inSockPos = pNode->getPortPos(true, inSock);
	const QPointF& outSockPos = m_nodes[outId]->getPortPos(false, outSock);

	pEdge->updatePos(outSockPos, inSockPos);
	addItem(pEdge);
	m_links[linkId] = pEdge;
	
    //outNode->toggleSocket(false, outSock, true);
}

void ZenoSubGraphScene::onLinkAboutToBeRemoved(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last)
{
	if (subGpIdx != m_subgIdx)
		return;

	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	QModelIndex linkIdx = pGraphsModel->linkIndex(first);
	Q_ASSERT(linkIdx.isValid());

    const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
    Q_ASSERT(m_links.find(linkId) != m_links.end());
    
    delete m_links[linkId];
	m_links.remove(linkId);
}

void ZenoSubGraphScene::onLinkRemoved(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last)
{
	if (subGpIdx != m_subgIdx)
		return;
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

    /*
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
    */
}

void ZenoSubGraphScene::paste(QPointF pos)
{
    /*
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
    */
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
    ZenoNode* pNode = nullptr;
    ZenoSocketItem* pSocket = nullptr;
    for (auto item : items)
    {
        if (pSocket = qgraphicsitem_cast<ZenoSocketItem*>(item))
            break;
    }
    if (!pSocket)
    {
        delete m_tempLink;
        m_tempLink = nullptr;
    }
    else
    {
        pNode = qgraphicsitem_cast<ZenoNode*>(pSocket->parentItem());

        //find zeno node.
        QString sockName;
        bool bInput = false;
        QPointF socketPos;
        pNode->getSocketInfoByItem(pSocket, sockName, socketPos, bInput);
        QString nodeid = pNode->nodeId();

        if (!m_tempLink && pSocket)
        {
            m_tempLink = new ZenoTempLink(nodeid, sockName, socketPos, bInput);
            addItem(m_tempLink);
            pSocket->toggle(true);
            return;
        }
        else if (m_tempLink && pSocket)
        {
            QString fixedNodeId, fixedSocket;
            bool fixedInput = false;
            QPointF fixedPos;
            m_tempLink->getFixedInfo(fixedNodeId, fixedSocket, fixedPos, fixedInput);

            if (fixedInput == bInput)
                return;

            QString outId, outPort, inId, inPort;
            QPointF outPos, inPos;
            if (fixedInput) {
                outId = nodeid;
                outPort = sockName;
                outPos = socketPos;
                inId = fixedNodeId;
                inPort = fixedSocket;
                inPos = fixedPos;
            }
            else {
                outId = fixedNodeId;
                outPort = fixedSocket;
                outPos = fixedPos;
                inId = nodeid;
                inPort = sockName;
                inPos = socketPos;
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
            bool fixedInput = false;
            QPointF fixedPos;
            QString fixedNodeId, fixedSocket;
            m_tempLink->getFixedInfo(fixedNodeId, fixedSocket, fixedPos, fixedInput);

            const QString& nodeId = pNode->nodeId();
            m_nodes[nodeId]->toggleSocket(fixedInput, sockName, false);

            removeItem(m_tempLink);
            delete m_tempLink;
            m_tempLink = nullptr;
            return;
        }
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
{//right click goes here
    if (subgIdx != m_subgIdx)
        return;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex idx = pGraphsModel->index(first, m_subgIdx);
    ZenoNode *pNode = createNode(idx, m_nodeParams);
    pNode->initUI(m_subgIdx, idx);
    addItem(pNode);
    QString id = pNode->nodeId();
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
        const INPUT_SOCKET inputSocket = pInputNode->inputParams()[sockName];
        for (QPersistentModelIndex index : inputSocket.linkIndice)
        {
            const QString& linkId = index.data(ROLE_OBJID).toString();
            m_links[linkId]->initDstPos(pos);
        }
    }
    else
    {
        ZenoNode* pOutputNode = m_nodes[nodeid];
        QPointF pos = pOutputNode->getPortPos(false, sockName);
        const OUTPUT_SOCKET outputSocket = pOutputNode->outputParams()[sockName];
        for (QPersistentModelIndex index : outputSocket.linkIndice)
        {
			const QString& linkId = index.data(ROLE_OBJID).toString();
			m_links[linkId]->initSrcPos(pos);
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
        const QPointF& inputPos = pNode->getPortPos(true, inSock);
        for (QPersistentModelIndex index : inputs[inSock].linkIndice)
        {
            const QString& linkId = index.data(ROLE_OBJID).toString();
            const QString& outNode = index.data(ROLE_OUTNODE).toString();
            const QString& outSock = index.data(ROLE_OUTSOCK).toString();

            const QPointF& outputPos = m_nodes[outNode]->getPortPos(false, outSock);

            ZenoFullLink* pLink = m_links[linkId];
			Q_ASSERT(pLink);
			pLink->updatePos(outputPos, inputPos);
        }
    }

    const OUTPUT_SOCKETS &outputs = pNode->outputParams();
    for (QString outputPort : outputs.keys())
    {
        const QPointF &outputPos = pNode->getPortPos(false, outputPort);

        for (QPersistentModelIndex index : outputs[outputPort].linkIndice)
        {
            const QString& linkId = index.data(ROLE_OBJID).toString();
            const QString& inNode = index.data(ROLE_INNODE).toString();
            const QString& inSock = index.data(ROLE_INSOCK).toString();

            QPointF inputPos = m_nodes[inNode]->getPortPos(true, inSock);
            ZenoFullLink* pLink = m_links[linkId];
            pLink->updatePos(outputPos, inputPos);
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
            pGraphsModel->removeLink(item->linkInfo(), m_subgIdx);
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
