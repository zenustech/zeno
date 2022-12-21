#include "zenosubgraphscene.h"
#include "zenonode.h"
#include "subnetnode.h"
#include "heatmapnode.h"
#include "cameranode.h"
#include "readfbxprim.h"
#include "zenolink.h"
#include <zenomodel/include/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/writer/zsgwriter.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/nodesys/nodesys_common.h>
#include <zenoui/nodesys/nodegrid.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zeno/utils/log.h>
#include "util/log.h"
#include "blackboardnode.h"
#include "acceptor/transferacceptor.h"
#include "variantptr.h"
#include <zenomodel/include/linkmodel.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenomodel/include/iparammodel.h>


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
        connect(pNode, &ZenoNode::socketClicked, this, &ZenoSubGraphScene::onSocketClicked);
        pNode->initUI(this, m_subgIdx, idx);
        addItem(pNode);
        const QString& nodeid = pNode->nodeId();
        m_nodes[nodeid] = pNode;
    }

    for (auto it : m_nodes)
    {
        ZenoNode *inNode = it.second;
        const QString& id = inNode->nodeId();
        const QModelIndex& idx = pGraphsModel->index(id, m_subgIdx);

        IParamModel* inputsModel = pGraphsModel->paramModel(idx, PARAM_INPUT);
        if (!inputsModel)
            continue;
        for (int r = 0; r < inputsModel->rowCount(); r++)
        {
            const QModelIndex& paramIdx = inputsModel->index(r, 0);
            const QString& inSock = paramIdx.data(ROLE_PARAM_NAME).toString();
            PARAM_LINKS links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            for (const QPersistentModelIndex& linkIdx : links)
            {
                const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
                const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
                const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
                const QModelIndex& inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
                ZenoNode* outNode = m_nodes[outId];
                ZASSERT_EXIT(outNode);

                ZenoFullLink* pEdge = new ZenoFullLink(linkIdx, outNode, inNode);
                addItem(pEdge);
                m_links[linkId] = pEdge;
                outNode->toggleSocket(false, outSock, true);
                outNode->getSocketItem(outSockIdx)->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
                inNode->toggleSocket(true, inSock, true);
                inNode->getSocketItem(inSockIdx)->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
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
    QAbstractItemModel* pLinkModel = pGraphsModel->linkModel();
    connect(pGraphsModel, &IGraphsModel::linkInserted, this, &ZenoSubGraphScene::onLinkInserted);
    connect(pLinkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &ZenoSubGraphScene::onLinkAboutToBeRemoved);
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
    else if (descName == "Blackboard")
    {
        return new BlackboardNode(params);
    }
    else if (descName == "CameraNode")
    {
        return new CameraNode(params);
    }
    else if(descName == "ReadFBXPrim")
    {
        return new ReadFBXPrim(params);
    }
    else
    {
        return new ZenoNode(params);
    }
}

void ZenoSubGraphScene::onZoomed(qreal factor)
{
    for (auto pair : m_nodes) {
        //pair.second->switchView(factor < 0.3);
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

void ZenoSubGraphScene::onLinkInserted(const QModelIndex& subGpIdx, const QModelIndex& parent, int first, int last)
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex linkIdx = pGraphsModel->linkIndex(first);
    viewAddLink(linkIdx);
}

void ZenoSubGraphScene::viewAddLink(const QModelIndex& linkIdx)
{
    const QString& linkId = linkIdx.data(ROLE_OBJID).toString();

    const QString& inId = linkIdx.data(ROLE_INNODE).toString();
    const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();
    const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
    const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

    if (m_nodes.find(inId) == m_nodes.end() || m_nodes.find(outId) == m_nodes.end())
    {
        //todo: half link across from two different subgraph.
        return;
    }

    ZenoNode* pInNode = m_nodes[inId];
    ZenoNode* pOutNode = m_nodes[outId];
    ZASSERT_EXIT(pInNode && pOutNode);

    ZenoFullLink* pEdge = new ZenoFullLink(QPersistentModelIndex(linkIdx), pOutNode, pInNode);
    addItem(pEdge);
    m_links[linkId] = pEdge;

    pInNode->onSocketLinkChanged(inSock, true, true);
    pOutNode->onSocketLinkChanged(outSock, false, true);
}

void ZenoSubGraphScene::onLinkAboutToBeRemoved(const QModelIndex&, int first, int last)
{
	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	QModelIndex linkIdx = pGraphsModel->linkIndex(first);
	ZASSERT_EXIT(linkIdx.isValid());
    viewRemoveLink(linkIdx);
}

void ZenoSubGraphScene::viewRemoveLink(const QModelIndex& linkIdx)
{
    const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
    if (m_links.find(linkId) == m_links.end())
        return;

    ZenoFullLink* pLink = m_links[linkId];
    m_links.remove(linkId);
    delete pLink;

    const QString& inId = linkIdx.data(ROLE_INNODE).toString();
    const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();
    const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
    const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();

    if (m_nodes.find(inId) != m_nodes.end())
        m_nodes[inId]->onSocketLinkChanged(inSock, true, false);
    if (m_nodes.find(outId) != m_nodes.end())
        m_nodes[outId]->onSocketLinkChanged(outSock, false, false);
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

QGraphicsItem* ZenoSubGraphScene::getNode(const QString& id)
{
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end(), nullptr);
    return m_nodes[id];
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
    //pNode->setSelected(true);
    m_errNodes.append(nodeid);
}

void ZenoSubGraphScene::clearMark()
{
    for (QString ident : m_errNodes)
    {
        if (m_nodes.find(ident) != m_nodes.end())
        {
            m_nodes[ident]->markError(false);
        }
    }
    m_errNodes.clear();
}

QList<ZenoParamWidget*> ZenoSubGraphScene::getScrollControls() const
{
    return m_scrollControls;
}

void ZenoSubGraphScene::addScrollControl(ZenoParamWidget* pWidget)
{
    if (!pWidget)
        return;
    m_scrollControls.append(pWidget);
    emit scrollControlAdded(pWidget);
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
            inSocket.info.links.clear();
            inSocket.info.nodeid = newId;
        }
        newNode[ROLE_INPUTS] = QVariant::fromValue(inputs);

        OUTPUT_SOCKETS outputs = newNode[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (OUTPUT_SOCKET& outSocket : outputs)
        {
            outSocket.info.links.clear();
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

            EdgeInfo newEdge(newOutNode, newInNode, outSock, inSock);

            inputSocket.info.links.append(newEdge);
            outputSocket.info.links.append(newEdge);

            //inputSocket.linkIndice.append(persistIdx);
            //outputSocket.linkIndice.append(persistIdx);

            inData[ROLE_INPUTS] = QVariant::fromValue(inputs);
            outData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
        }
    }

    ZsgWriter::getInstance().dumpToClipboard(newNodes);
}

void ZenoSubGraphScene::paste(QPointF pos)
{
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (pMimeData->hasText() && pGraphsModel)
    {
        const QString& strJson = pMimeData->text();
        TransferAcceptor acceptor(pGraphsModel);
        ZsgReader::getInstance().importNodes(pGraphsModel, m_subgIdx, strJson, pos, &acceptor);
        acceptor.reAllocIdents();

        QMap<QString, NODE_DATA> nodes;
        QList<EdgeInfo> links;
        acceptor.getDumpData(nodes, links);
        //todo: ret value for api.
        pGraphsModel->importNodes(nodes, links, pos, m_subgIdx, true);

        //mark selection for all nodes and links.
        clearSelection();
        for (QString ident : nodes.keys())
        {
            ZASSERT_EXIT(m_nodes.find(ident) != m_nodes.end());
            m_nodes[ident]->setSelected(true);
        }
    }
}

void ZenoSubGraphScene::reload(const QModelIndex& subGpIdx)
{
    if (subGpIdx != m_subgIdx)
    {
        clear();
        initModel(subGpIdx);
    }
}

void ZenoSubGraphScene::onSocketClicked(ZenoSocketItem* pSocketItem)
{
    ZASSERT_EXIT(pSocketItem);

    QModelIndex paramIdx = pSocketItem->paramIndex();
    ZASSERT_EXIT(paramIdx.isValid());

    if (!paramIdx.data(ROLE_VPARAM_IS_COREPARAM).toBool())
    {
        QMessageBox::information(nullptr, "Error", "cannot generate link from custom socket now");
        return;
    }

    bool bInput = pSocketItem->isInputSocket();
    QString nodeid = pSocketItem->nodeIdent();

    PARAM_CONTROL ctrl = (PARAM_CONTROL)paramIdx.data(ROLE_PARAM_CTRL).toInt();
    SOCKET_PROPERTY prop = (SOCKET_PROPERTY)paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();
    QPointF socketPos = pSocketItem->center();

    ZASSERT_EXIT(m_nodes.find(nodeid) != m_nodes.end());
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    PARAM_LINKS linkIndice = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
    bool bDisconnetLink = prop != SOCKPROP_MULTILINK && bInput && !linkIndice.isEmpty();
    if (bDisconnetLink)
    {
        //todo: multiple link
        QPersistentModelIndex linkIdx = linkIndice[0];

        //disconnect the old link.
        const QString& outNode = linkIdx.data(ROLE_OUTNODE).toString();
        const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
        const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

        //remove current link at view.
        viewRemoveLink(linkIdx);

        socketPos = m_nodes[outNode]->getSocketPos(outSockIdx);
        pSocketItem = m_nodes[outNode]->getSocketItem(outSockIdx);
        m_tempLink = new ZenoTempLink(pSocketItem, outNode, socketPos, false);
        m_tempLink->setOldLink(linkIdx);
        addItem(m_tempLink);
    }
    else
    {
        m_tempLink = new ZenoTempLink(pSocketItem, nodeid, socketPos, bInput);
        addItem(m_tempLink);
    }
}

void ZenoSubGraphScene::onSocketAbsorted(const QPointF mousePos)
{
    bool bFixedInput = false;
    QString nodeId;
    QPointF fixedPos;
    m_tempLink->getFixedInfo(nodeId, fixedPos, bFixedInput);

    QPointF pos = mousePos;
    QList<QGraphicsItem *> catchedItems = items(pos);
    QList<ZenoNode *> catchNodes;
    QList<ZenoSocketItem* > catchSocks;
    for (QGraphicsItem *item : catchedItems)
    {
        if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
        {
            if (pNode->index().data(ROLE_OBJID).toString() != nodeId)
            {
                catchNodes.append(pNode);
            }
        }
        else if (ZenoSocketItem* sock = qgraphicsitem_cast<ZenoSocketItem*>(item))
        {
            if (sock->isEnabled())
                catchSocks.append(sock);
        }
    }
    //adsorption
    if (!catchSocks.isEmpty())
    {
        ZenoSocketItem* pSocket = catchSocks[0];
        if (pSocket)
        {
            bool bInput = pSocket->isInputSocket();
            QString nodeid2 = pSocket->nodeIdent();
            if (bInput != bFixedInput && nodeid2 != nodeId)
            {
                pos = pSocket->center();
                m_tempLink->setAdsortedSocket(pSocket);
                m_tempLink->setFloatingPos(pos);
                return;
            }
        }
    }
    
    if (!catchNodes.isEmpty())
    {
        ZenoNode *pTarget = nullptr;
        float minDist = std::numeric_limits<float>::max();
        for (ZenoNode *pNode : catchNodes)
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
        ZenoSocketItem *pSocket = pTarget->getNearestSocket(pos, !bFixedInput);
        if (pSocket)
        {
            pos = pSocket->center();
        }
        m_tempLink->setAdsortedSocket(pSocket);
        m_tempLink->setFloatingPos(pos);
    }
    else
    {
        m_tempLink->setAdsortedSocket(nullptr);
        m_tempLink->setFloatingPos(pos);
    }
}

void ZenoSubGraphScene::onTempLinkClosed()
{
    if (!m_tempLink)
        return;

    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (ZenoSocketItem* targetSock = m_tempLink->getAdsorbedSocket())
    {
        if (!targetSock->paramIndex().data(ROLE_VPARAM_IS_COREPARAM).toBool())
        {
            QMessageBox::information(nullptr, "Error", "cannot generate link from custom socket now");
            return;
        }

        IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);

        bool bTargetInput = targetSock->isInputSocket();

        QString fixedNodeId;
        bool fixedInput = false;
        QPointF fixedPos;
        m_tempLink->getFixedInfo(fixedNodeId, fixedPos, fixedInput);

        if (bTargetInput != fixedInput)
        {
            QPersistentModelIndex fromSockIdx, toSockIdx;
            if (fixedInput) {
                fromSockIdx = targetSock->paramIndex();
                toSockIdx = m_tempLink->getFixedSocket()->paramIndex();
            } else {
                fromSockIdx = m_tempLink->getFixedSocket()->paramIndex();
                toSockIdx = targetSock->paramIndex();
            }

            const QPersistentModelIndex& oldLink = m_tempLink->oldLink();
            if (oldLink.isValid())
            {
                //same link?
                if (oldLink.data(ROLE_OUTSOCK_IDX).toModelIndex() == fromSockIdx &&
                    oldLink.data(ROLE_INSOCK_IDX).toModelIndex() == toSockIdx)
                {
                    viewAddLink(oldLink);
                    return;
                }
            }

            pGraphsModel->beginTransaction(tr("add Link"));
            zeno::scope_exit sp([=]() { pGraphsModel->endTransaction(); });

            //remove the old Link first.
            if (oldLink.isValid())
                pGraphsModel->removeLink(oldLink, true);

            //dict panel.
            SOCKET_PROPERTY inProp = (SOCKET_PROPERTY)toSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
            if (bTargetInput && (inProp & SOCKPROP_DICTPANEL))
            {
                SOCKET_PROPERTY outProp = (SOCKET_PROPERTY)fromSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
                QAbstractItemModel *pKeyObjModel =
                    QVariantPtr<QAbstractItemModel>::asPtr(toSockIdx.data(ROLE_VPARAM_LINK_MODEL));

                //check if outSock is a dict
                if (outProp & SOCKPROP_DICTPANEL)
                {
                    //legacy dict connection, and then we have to remove all inner dict key connection.
                    ZASSERT_EXIT(pKeyObjModel);
                    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
                    {
                        const QModelIndex& keyIdx = pKeyObjModel->index(r, 0);
                        QPersistentModelIndex _linkIdx  = keyIdx.data(ROLE_LINK_IDX).toPersistentModelIndex();
                        pGraphsModel->removeLink(_linkIdx);
                    }
                }
                else
                {
                    // link to inner dict key automatically.
                    int n = pKeyObjModel->rowCount();
                    pKeyObjModel->insertRow(n);
                    toSockIdx = pKeyObjModel->index(n, 0);
                }
            }

            //remove the edge in inNode:inSock, if exists.
            if (bTargetInput && inProp != SOCKPROP_MULTILINK)
            {
                QPersistentModelIndex linkIdx;
                const QModelIndex& paramIdx = targetSock->paramIndex();
                const PARAM_LINKS& links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                if (!links.isEmpty())
                    linkIdx = links[0];
                if (linkIdx.isValid())
                    pGraphsModel->removeLink(linkIdx, true);
            }

            pGraphsModel->addLink(fromSockIdx, toSockIdx, true);
            return;
        }
    }

    const QPersistentModelIndex& oldLink = m_tempLink->oldLink();
    if (oldLink.isValid())
    {
        pGraphsModel->removeLink(oldLink, true);
    }
}

void ZenoSubGraphScene::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsScene::mousePressEvent(event);
}

void ZenoSubGraphScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_tempLink)
    {
        onSocketAbsorted(event->scenePos());
        return;
    }
    QGraphicsScene::mouseMoveEvent(event);
}

void ZenoSubGraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_tempLink && event->button() != Qt::MidButton && event->button() != Qt::RightButton)
    {
        onTempLinkClosed();
        removeItem(m_tempLink);
        delete m_tempLink;
        m_tempLink = nullptr;
        return;
    }
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
    connect(pNode, &ZenoNode::socketClicked, this, &ZenoSubGraphScene::onSocketClicked);
    pNode->initUI(this, m_subgIdx, idx);
    addItem(pNode);
    QString id = pNode->nodeId();
    m_nodes[id] = pNode;
}

void ZenoSubGraphScene::keyPressEvent(QKeyEvent* event)
{
    QGraphicsScene::keyPressEvent(event);
    if (!event->isAccepted() && event->key() == Qt::Key_Delete)
    {
        if (m_tempLink)
        {
            removeItem(m_tempLink);
            delete m_tempLink;
            m_tempLink = nullptr;
        }
        else
        {
            QList<QGraphicsItem*> selItems = this->selectedItems();
            QList<QPersistentModelIndex> nodes;
            QList<QPersistentModelIndex> links;
            for (auto item : selItems)
            {
                if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
                {
                    nodes.append(pNode->index());
                }
                else if (ZenoFullLink* pLink = qgraphicsitem_cast<ZenoFullLink*>(item))
                {
                    links.append(pLink->linkInfo());
                }
            }
            if (!nodes.isEmpty() || !links.isEmpty())
            {
                IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pGraphsModel);

                pGraphsModel->beginTransaction("remove nodes and links");
                for (const QModelIndex &linkIdx : links)
                {
                    pGraphsModel->removeLink(linkIdx, true);
                }
                for (const QModelIndex &nodeIdx : nodes)
                {
                    QString id = nodeIdx.data(ROLE_OBJID).toString();
                    pGraphsModel->removeNode(id, m_subgIdx, true);
                }
                pGraphsModel->endTransaction();
            }
        }
    }
}
