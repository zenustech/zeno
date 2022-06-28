#include "zenosubgraphscene.h"
#include "zenonode.h"
#include "subnetnode.h"
#include "heatmapnode.h"
#include "curvenode.h"
#include "dynamicnumbernode.h"
#include "zenolink.h"
#include <zenoui/model/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/writer/zsgwriter.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/nodesys/nodegrid.h>
#include <zenoui/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zeno/utils/log.h>
#include "util/log.h"
#include "makelistnode.h"
#include "blackboardnode.h"
#include "acceptor/modelacceptor.h"


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
    ZASSERT_EXIT(pGraphsModel);

    disconnect(pGraphsModel, SIGNAL(reloaded(const QModelIndex&)), this, SLOT(reload(const QModelIndex&)));
    disconnect(pGraphsModel, SIGNAL(clearLayout(const QModelIndex&)), this, SLOT(clearLayout(const QModelIndex&)));

    //todo: better to connect _dataChanged to global managment, and dispatch to specify scene.
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
    }

    for (auto it : m_nodes)
    {
        ZenoNode *inNode = it.second;
        const QString& id = inNode->nodeId();
        const INPUT_SOCKETS& inputs = inNode->inputParams();
        for (QString inSock : inputs.keys())
        {
            QPointF inSockPos = inNode->getPortPos(true, inSock);
            const INPUT_SOCKET& inputSocket = inputs[inSock];
            for (const QPersistentModelIndex& linkIdx : inputSocket.linkIndice)
            {
                const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
                const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
                const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                ZenoNode* outNode = m_nodes[outId];
                const QPointF& outSockPos = outNode->getPortPos(false, outSock);

                ZenoFullLink *pEdge = new ZenoFullLink(linkIdx, outNode, inNode);
                addItem(pEdge);
                m_links[linkId] = pEdge;
                outNode->toggleSocket(false, outSock, true);
                outNode->getSocketItem(false, outSock)->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
                inNode->toggleSocket(true, inSock, true);
                inNode->getSocketItem(true, inSock)->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
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
    const QString& descName = idx.data(ROLE_OBJNAME).toString();
    if (descName == "SubInput")
    {
        return new SubnetNode(true, params);
    }
    else if (descName == "SubOutput")
    {
        return new SubnetNode(false, params);
    }
    else if (descName == "MakeHeatmap")
    {
        return new MakeHeatMapNode(params);
    }
    else if (descName == "MakeCurve")
    {
        return new MakeCurveNode(params);
    }
    else if (descName == "DynamicNumber")
    {
        return new DynamicNumberNode(params);
    }
    else if (descName == "MakeList")
    {
        return new MakeListNode(params);
    }
    else if (descName == "Blackboard")
    {
        return new BlackboardNode(params);
    }
    else
    {
        return new ZenoNode(params);
    }
}

void ZenoSubGraphScene::onZoomed(qreal factor)
{
    for (auto pair : m_nodes) {
        pair.second->switchView(factor < 0.3);
    }
}

void ZenoSubGraphScene::onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
    if (subGpIdx != m_subgIdx)
        return;

	QString id = idx.data(ROLE_OBJID).toString();

    if (role == ROLE_OBJPOS)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
	    QPointF pos = idx.data(ROLE_OBJPOS).toPointF();
        m_nodes[id]->setPos(pos);
	}
    if (role == ROLE_INPUTS || role == ROLE_OUTPUTS)
    {
	    //it's diffcult to detect which input/output socket has changed on this method.
        //unless:
        //1. add a member to INPUT_SOCKS/OUTPUT_SOCKS, to specify which key had changed.
        //2. update all control value associated with input socket anyway.
        //
        //now we choose the second.
        if (m_nodes.find(id) != m_nodes.end())
        {
            m_nodes[id]->onSocketsUpdate(role == ROLE_INPUTS);
        }
	}
    if (role == ROLE_OPTIONS)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        int options = idx.data(ROLE_OPTIONS).toInt();
        m_nodes[id]->onOptionsUpdated(options);
	}
	if (role == ROLE_COLLASPED)
	{
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
		bool bCollasped = idx.data(ROLE_COLLASPED).toBool();
		m_nodes[id]->onCollaspeUpdated(bCollasped);
	}
    if (role == ROLE_MODIFY_PARAM)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        PARAM_INFO info = idx.data(ROLE_MODIFY_PARAM).value<PARAM_INFO>();
        m_nodes[id]->onParamUpdated(info.name, info.value);
    }
    if (role == ROLE_OBJNAME)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        m_nodes[id]->onNameUpdated(idx.data(ROLE_OBJNAME).toString());
    }
    if (role == ROLE_PARAMS_NO_DESC)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        m_nodes[id]->onUpdateParamsNotDesc();
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

	const QString& linkId = linkIdx.data(ROLE_OBJID).toString();

    const QString& inId = linkIdx.data(ROLE_INNODE).toString();
    const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();
	const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
	const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

    ZenoNode* pNode = m_nodes[inId];
    const QPointF& inSockPos = pNode->getPortPos(true, inSock);
	const QPointF& outSockPos = m_nodes[outId]->getPortPos(false, outSock);

    ZenoFullLink *pEdge = new ZenoFullLink(QPersistentModelIndex(linkIdx), m_nodes[outId], m_nodes[inId]);
	addItem(pEdge);
	m_links[linkId] = pEdge;

    pNode->onSocketLinkChanged(inSock, true, true);
    m_nodes[outId]->onSocketLinkChanged(outSock, false, true);
}

void ZenoSubGraphScene::onLinkAboutToBeRemoved(const QModelIndex& subGpIdx, const QModelIndex&, int first, int last)
{
	if (subGpIdx != m_subgIdx)
		return;

	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	QModelIndex linkIdx = pGraphsModel->linkIndex(first);
	ZASSERT_EXIT(linkIdx.isValid());

    const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
    ZASSERT_EXIT(m_links.find(linkId) != m_links.end());

    delete m_links[linkId];
	m_links.remove(linkId);

	const QString& inId = linkIdx.data(ROLE_INNODE).toString();
	const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();
	const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
	const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

    if (m_nodes.find(inId) != m_nodes.end())
	    m_nodes[inId]->onSocketLinkChanged(inSock, true, false);
    if (m_nodes.find(outId) != m_nodes.end())
	    m_nodes[outId]->onSocketLinkChanged(outSock, false, false);
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
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id]->setSelected(true);
}

void ZenoSubGraphScene::markError(const QString& nodeid)
{
    ZASSERT_EXIT(m_nodes.find(nodeid) != m_nodes.end());
    ZenoNode *pNode = m_nodes[nodeid];
    pNode->markError(true);
    pNode->setSelected(true);
    m_errNodes.append(pNode);
}

void ZenoSubGraphScene::clearMark()
{
    for (ZenoNode* pNode : m_errNodes)
    {
        pNode->markError(false);
    }
}

void ZenoSubGraphScene::undo()
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    pGraphsModel->undo();
}

void ZenoSubGraphScene::redo()
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    pGraphsModel->redo();
}

QModelIndexList ZenoSubGraphScene::selectNodesIndice() const
{
    QModelIndexList nodesIndice;
    QList<QGraphicsItem *> selItems = selectedItems();
    for (auto item : selItems)
    {
        if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
        {
            nodesIndice.append(pNode->index());
        }
    }
    return nodesIndice;
}

void ZenoSubGraphScene::copy()
{
    copy2();
    /* legacy copy by custom mimedata.
    QList<QGraphicsItem*> selItems = this->selectedItems();
    if (selItems.isEmpty())
        return;

    QMap<QString, ZenoNode*> selNodes;

    QModelIndexList nodesIndice;
    for (auto item : selItems)
    {
        if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
        {
            nodesIndice.append(pNode->index());
            selNodes.insert(pNode->nodeId(), pNode);
        }
    }

    if (selNodes.isEmpty())
    {
        QApplication::clipboard()->clear();
    }

    NODES_MIME_DATA* pNodesData = new NODES_MIME_DATA;
    pNodesData->nodes = nodesIndice;
    pNodesData->m_fromSubg = m_subgIdx;

    QMimeData *pMimeData = new QMimeData;
    pMimeData->setUserData(MINETYPE_MULTI_NODES, pNodesData);
    QApplication::clipboard()->setMimeData(pMimeData);
    */
}

void ZenoSubGraphScene::copy2()
{
    QList<QGraphicsItem*> selItems = this->selectedItems();
    if (selItems.isEmpty())
        return;

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    //first record all nodes.
    QModelIndexList selNodes, selLinks;
    for (auto item : selItems)
    {
        if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
        {
            selNodes.append(pNode->index());
        }
        else if (ZenoFullLink *pLink = qgraphicsitem_cast<ZenoFullLink*>(item))
        {
            selLinks.append(pLink->linkInfo());
        }
    }

    QMap<QString, NODE_DATA> oldNodes, newNodes;
    QMap<QString, QString> old2new, new2old;

    for (QModelIndex idx : selNodes)
    {
        NODE_DATA old = pModel->itemData(idx, m_subgIdx);
        const QString& oldId = old[ROLE_OBJID].toString();
        oldNodes.insert(oldId, old);

        NODE_DATA newNode = old;
        const QString& nodeName = old[ROLE_OBJNAME].toString();
        const QString& newId = UiHelper::generateUuid(nodeName);
        newNode[ROLE_OBJID] = newId;
        newNode[ROLE_OBJPOS] = old[ROLE_OBJPOS];

        //clear all link info in socket.
        INPUT_SOCKETS inputs = newNode[ROLE_INPUTS].value<INPUT_SOCKETS>();
        for (INPUT_SOCKET& inSocket : inputs)
        {
            inSocket.linkIndice.clear();
            inSocket.outNodes.clear();
            inSocket.info.nodeid = newId;
        }
        newNode[ROLE_INPUTS] = QVariant::fromValue(inputs);

        OUTPUT_SOCKETS outputs = newNode[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (OUTPUT_SOCKET& outSocket : outputs)
        {
            outSocket.linkIndice.clear();
            outSocket.linkIndice.clear();
            outSocket.info.nodeid = newId;
        }
        newNode[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

        newNodes.insert(newId, newNode);

        old2new.insert(oldId, newId);
        new2old.insert(newId, oldId);
    }

    QStandardItemModel tempLinkModel;
    //copy all link.
    for (QModelIndex idx : selLinks)
    {
        const QString& outNode = idx.data(ROLE_OUTNODE).toString();
        const QString& outSock = idx.data(ROLE_OUTSOCK).toString();
        const QString& inNode = idx.data(ROLE_INNODE).toString();
        const QString& inSock = idx.data(ROLE_INSOCK).toString();

        if (oldNodes.find(outNode) != oldNodes.end() &&
            oldNodes.find(inNode) != oldNodes.end())
        {
            const QString& newOutNode = old2new[outNode];
            const QString& newInNode = old2new[inNode];
            NODE_DATA& inData = newNodes[newInNode];
            NODE_DATA& outData = newNodes[newOutNode];
            INPUT_SOCKETS inputs = inData[ROLE_INPUTS].value<INPUT_SOCKETS>();
            OUTPUT_SOCKETS outputs = outData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

            ZASSERT_EXIT(inputs.find(inSock) != inputs.end());
            ZASSERT_EXIT(outputs.find(outSock) != outputs.end());
            INPUT_SOCKET& inputSocket = inputs[inSock];
            OUTPUT_SOCKET& outputSocket = outputs[outSock];

            //construct new link.
            QStandardItem* pItem = new QStandardItem;
            pItem->setData(UiHelper::generateUuid(), ROLE_OBJID);
            pItem->setData(newInNode, ROLE_INNODE);
            pItem->setData(inSock, ROLE_INSOCK);
            pItem->setData(newOutNode, ROLE_OUTNODE);
            pItem->setData(outSock, ROLE_OUTSOCK);
            tempLinkModel.appendRow(pItem);
            QModelIndex linkIdx = tempLinkModel.indexFromItem(pItem);
            QPersistentModelIndex persistIdx(linkIdx);

            inputSocket.linkIndice.append(persistIdx);
            outputSocket.linkIndice.append(persistIdx);

            inData[ROLE_INPUTS] = QVariant::fromValue(inputs);
            outData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
        }
    }

    ZsgWriter::getInstance().dumpToClipboard(newNodes);
}

void ZenoSubGraphScene::paste(QPointF pos)
{
    /*
    * base custom mime data.
    * 
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (QObjectUserData *pUserData = pMimeData->userData(MINETYPE_MULTI_NODES))
    {
        NODES_MIME_DATA* pNodesData = static_cast<NODES_MIME_DATA*>(pUserData);
        if (pNodesData->nodes.isEmpty())
            return;
        pGraphsModel->copyPaste(pNodesData->m_fromSubg, pNodesData->nodes, m_subgIdx, pos, true);
        clearSelection();
        //todo: select them
    }
    */
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (pMimeData->hasText())
    {
        const QString& strJson = pMimeData->text();
        ModelAcceptor acceptor(nullptr, false);
        ZsgReader::getInstance().importNodes(pGraphsModel, m_subgIdx, strJson, pos, &acceptor);
    }
}

QPointF ZenoSubGraphScene::getSocketPos(bool bInput, const QString &nodeid, const QString &portName)
{
    auto it = m_nodes.find(nodeid);
    ZASSERT_EXIT(it != m_nodes.end(), QPointF());
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
    if (event->button() == Qt::LeftButton)
    {
		QList<QGraphicsItem*> items = this->items(event->scenePos());
		ZenoNode* pNode = nullptr;
		ZenoSocketItem* pSocket = nullptr;
		for (auto item : items)
		{
			if (pSocket = qgraphicsitem_cast<ZenoSocketItem*>(item))
				break;
		}
        if (m_tempLink && !pSocket)
        {
            pSocket = m_tempLink->getAdsorbedSocket();
        }

		if (!pSocket)
		{
			delete m_tempLink;
			m_tempLink = nullptr;
		}
		else
		{
			pNode = qgraphicsitem_cast<ZenoNode*>(pSocket->parentItem());
			const QString& nodeid = pNode->nodeId();
			IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

			QString sockName;
			bool bInput = false;
			QPointF socketPos;
			QPersistentModelIndex linkIdx;
			pNode->getSocketInfoByItem(pSocket, sockName, socketPos, bInput, linkIdx);

			if (!m_tempLink)
			{
				if (linkIdx.isValid() && bInput)
				{
					//disconnect the old link.
					const QString& outNode = linkIdx.data(ROLE_OUTNODE).toString();
					const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

					pGraphsModel->removeLink(linkIdx, m_subgIdx, true);

					socketPos = m_nodes[outNode]->getPortPos(false, outSock);
                    ZenoSocketItem* pSocketItem = m_nodes[outNode]->getSocketItem(false, outSock);
                    m_tempLink = new ZenoTempLink(pSocketItem, outNode, outSock, socketPos, false);
					addItem(m_tempLink);
				}
				else
				{
                    ZenoSocketItem* pSocketItem = m_nodes[nodeid]->getSocketItem(bInput, sockName);
                    m_tempLink = new ZenoTempLink(pSocketItem, nodeid, sockName, socketPos, bInput);
					addItem(m_tempLink);
				}
				return;
			}
			else if (m_tempLink)
			{
				QString fixedNodeId, fixedSocket;
				bool fixedInput = false;
				QPointF fixedPos;
				m_tempLink->getFixedInfo(fixedNodeId, fixedSocket, fixedPos, fixedInput);

				if (fixedInput == bInput)
					return;

				QString outNode, outSock, inNode, inSock;
				QPointF outPos, inPos;
				if (fixedInput) {
					outNode = nodeid;
					outSock = sockName;
					outPos = socketPos;
					inNode = fixedNodeId;
					inSock = fixedSocket;
					inPos = fixedPos;
				}
				else {
					outNode = fixedNodeId;
					outSock = fixedSocket;
					outPos = fixedPos;
					inNode = nodeid;
					inSock = sockName;
					inPos = socketPos;
				}

				//remove the edge in inNode:inSock, if exists.
				if (bInput)
				{
					QPersistentModelIndex linkIdx;
					m_nodes[inNode]->getSocketInfoByItem(pSocket, sockName, socketPos, bInput, linkIdx);
					if (linkIdx.isValid())
						pGraphsModel->removeLink(linkIdx, m_subgIdx, true);
				}

				EdgeInfo info(outNode, inNode, outSock, inSock);
				IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
				pGraphsModel->addLink(info, m_subgIdx, true);

				removeItem(m_tempLink);
				delete m_tempLink;
				m_tempLink = nullptr;
				return;
			}
		}
    }
    QGraphicsScene::mousePressEvent(event);
}

void ZenoSubGraphScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_tempLink)
    {
        bool bFixedInput = false;
        QString nodeId, sockName;
        QPointF fixedPos;
        m_tempLink->getFixedInfo(nodeId, sockName, fixedPos, bFixedInput);

        QPointF pos = event->scenePos();
        QList<QGraphicsItem*> catchedItems = items(pos);
        QList<ZenoNode*> catchNodes;
        for (QGraphicsItem* item : catchedItems)
        {
            if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
            {
                if (pNode->index().data(ROLE_OBJID).toString() != nodeId)
                {
                    catchNodes.append(pNode);
                }
            }
        }
        //adsorption
        if (!catchNodes.isEmpty())
        {
            ZenoNode *pTarget = nullptr;
            float minDist = std::numeric_limits<float>::max();
            for (ZenoNode* pNode : catchNodes)
            {
                QPointF nodePos = pNode->sceneBoundingRect().center();
                QPointF offset = nodePos - pos;
                float dist = std::sqrt(offset.x() * offset.x() + offset.y() * offset.y());
                if (dist < minDist)
                {
                    minDist = dist;
                    pTarget = pNode;
                }
            }
            ZASSERT_EXIT(pTarget);

            minDist = std::numeric_limits<float>::max();

            //find the min dist from socket to current pos.
            ZenoSocketItem* pSocket = pTarget->getNearestSocket(pos, !bFixedInput);
            if (pSocket) {
                pos = pSocket->sceneBoundingRect().center();
            }
            m_tempLink->setAdsortedSocket(pSocket);
        }
        else {
            m_tempLink->setAdsortedSocket(nullptr);
        }
        m_tempLink->setFloatingPos(pos);
    }
    QGraphicsScene::mouseMoveEvent(event);
}

void ZenoSubGraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsScene::mouseReleaseEvent(event);
}

void ZenoSubGraphScene::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    //send to ZenoNode.
    QGraphicsScene::contextMenuEvent(event);
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
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
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
            ZASSERT_EXIT(pLink);
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
        }
    }
}

void ZenoSubGraphScene::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        QList<QGraphicsItem*> selItems = this->selectedItems();
        QList<QPersistentModelIndex> nodes;
        QList<QPersistentModelIndex> links;
        for (auto item : selItems)
        {
            if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
            {
                nodes.append(pNode->index());
            }
            else if (ZenoFullLink *pLink = qgraphicsitem_cast<ZenoFullLink *>(item))
            {
                links.append(pLink->linkInfo());
            }
        }
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);
        pGraphsModel->removeNodeLinks(nodes, links, m_subgIdx);

    }
    QGraphicsScene::keyPressEvent(event);
}
