#include "zenosubgraphscene.h"
#include "zenonode.h"
#include "subnetnode.h"
#include "heatmapnode.h"
#include "cameranode.h"
#include "pythonnode.h"
#include "commandnode.h"
#include "readfbxprim.h"
#include "livenode.h"
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
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/vparamitem.h>
#include <zenomodel/include/command.h>
#include "nodesys/groupnode.h"
#include <zenoui/style/zenostyle.h>
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "zenomainwindow.h"
#include <zeno/extra/ObjectsManager.h>
#include <viewportinteraction/picker.h>
#include "settings/zenosettingsmanager.h"
#include "timeline/ztimeline.h"
#include <zenoui/comctrl/gv/zgraphicsnetlabel.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include "nodesys/pythonmaterialnode.h"

ZenoSubGraphScene::ZenoSubGraphScene(QObject *parent)
    : QGraphicsScene(parent)
    , m_tempLink(nullptr)
    , m_bOnceOn(false)
    , m_bBypassOn(false)
    , m_bViewOn(false)
    , m_bCacheOn(false)
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

    QVector<ZenoNode *> blackboardVect;
    for (int r = 0; r < pGraphsModel->itemCount(m_subgIdx); r++)
    {
        const QModelIndex& idx = pGraphsModel->index(r, m_subgIdx);
        ZenoNode* pNode = createNode(idx, m_nodeParams);
        connect(pNode, &ZenoNode::socketClicked, this, &ZenoSubGraphScene::onSocketClicked);
        connect(pNode, &ZenoNode::nodePosChangedSignal, this, &ZenoSubGraphScene::onNodePosChanged);
        pNode->initUI(this, m_subgIdx, idx);
        addItem(pNode);
        const QString& nodeid = pNode->nodeId();
        m_nodes[nodeid] = pNode;
        if (pNode->nodeName() == "Group") 
        {
            blackboardVect << pNode;
        }
        if (m_subgIdx.data(ROLE_OBJNAME).toString() == "main") {
            pGraphsModel->markNodeDataChanged(idx, false);
            pNode->onMarkDataChanged(true);
        }
    }

    for (auto it : m_nodes)
    {
        ZenoNode *inNode = it.second;
        const QString& id = inNode->nodeId();
        const QModelIndex& idx = pGraphsModel->index(id, m_subgIdx);

        NodeParamModel* viewParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
        const QModelIndexList& lst = viewParams->getInputIndice();
        for (int r = 0; r < lst.size(); r++)
        {
            const QModelIndex& paramIdx = lst[r];
            const QString& inSock = paramIdx.data(ROLE_PARAM_NAME).toString();
            const int inSockProp = paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();

            PARAM_LINKS links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
            {
                for (const QPersistentModelIndex& linkIdx : links)
                {
                    initLink(linkIdx);
                }
            }
            else
            {
                if (inSockProp & SOCKPROP_DICTLIST_PANEL)
                {
                    QAbstractItemModel* pKeyObjModel = QVariantPtr<QAbstractItemModel>::asPtr(paramIdx.data(ROLE_VPARAM_LINK_MODEL));
                    for (int _r = 0; _r < pKeyObjModel->rowCount(); _r++)
                    {
                        const QModelIndex& keyIdx = pKeyObjModel->index(_r, 0);
                        ZASSERT_EXIT(keyIdx.isValid());
                        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                        if (!links.isEmpty())
                        {
                            const QModelIndex& linkIdx = links[0];
                            initLink(linkIdx);
                        }
                    }
                }
            }
        }

        for (auto node : blackboardVect)
        {
            PARAMS_INFO params = node->index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
            BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
            if (info.items.contains(id) && qobject_cast<GroupNode*>(node))
            {
                GroupNode *pGroupNode = qobject_cast<GroupNode *>(node);
                if (pGroupNode)
                    pGroupNode->appendChildItem(inNode);
                else
                    inNode->setGroupNode(pGroupNode);
            }
        }
    }

    LinkModel* pLegacyLinks = pGraphsModel->legacyLinks(m_subgIdx);
    if (pLegacyLinks)
    {
        for (int r = 0; r < pLegacyLinks->rowCount(); r++)
        {
            QModelIndex linkIdx = pLegacyLinks->index(r, 0);
            initLink(linkIdx);
        }
    }

    //a more effecient way is collect scene togther and send msg to specific scene.
	connect(pGraphsModel, SIGNAL(reloaded(const QModelIndex&)), this, SLOT(reload(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(clearLayout(const QModelIndex&)), this, SLOT(clearLayout(const QModelIndex&)));
    connect(pGraphsModel, SIGNAL(_dataChanged(const QModelIndex&, const QModelIndex&, int)), this, SLOT(onDataChanged(const QModelIndex&, const QModelIndex&, int)));
    connect(pGraphsModel, SIGNAL(_rowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, const QModelIndex&, int, int)));
    connect(pGraphsModel, SIGNAL(_rowsInserted(const QModelIndex&, const QModelIndex&, int, int)), this, SLOT(onRowsInserted(const QModelIndex&, const QModelIndex&, int, int)));

    //link sync:
    QAbstractItemModel* pLinkModel = pGraphsModel->linkModel(m_subgIdx);
    connect(pLinkModel, &QAbstractItemModel::rowsInserted, this, &ZenoSubGraphScene::onLinkInserted);
    connect(pLinkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &ZenoSubGraphScene::onLinkAboutToBeRemoved);
}

void ZenoSubGraphScene::initLink(const QModelIndex& linkIdx)
{
    if (!linkIdx.isValid())
        return;

    const QString& linkId = linkIdx.data(ROLE_OBJID).toString();
    if (m_links.find(linkId) != m_links.end())
        return;

    const QString& inId = linkIdx.data(ROLE_INNODE).toString();
    const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
    const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    const QModelIndex& inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();

    ZenoNode* inNode = m_nodes[inId];
    ZenoNode* outNode = m_nodes[outId];
    ZASSERT_EXIT(inNode && outNode);

    ZenoFullLink *pEdge = new ZenoFullLink(linkIdx, outNode, inNode);
    addItem(pEdge);
    m_links[linkId] = pEdge;

    ZenoSocketItem *socketItem = outNode->getSocketItem(outSockIdx);
    ZASSERT_EXIT(socketItem);
    socketItem->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
    //socketItem->toggle(true);

    socketItem = inNode->getSocketItem(inSockIdx);
    ZASSERT_EXIT(socketItem);
    socketItem->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
    //socketItem->toggle(true);
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
    else if (descName == "Group")
    {
        return new GroupNode(params);
    }
    else if (descName == "CameraNode")
    {
        return new CameraNode(params, 0);
    }
    else if (descName == "LightNode")
    {
        return new LightNode(params, 0);
    }
    else if (descName == "MakeCamera")
    {
        return new CameraNode(params, 1);
    }
    else if(descName == "ReadFBXPrim")
    {
        return new ReadFBXPrim(params);
    }
    else if(descName == "LiveMeshNode")
    {
        return new LiveMeshNode(params);
    }
    else if(descName == "EvalBlenderFile")
    {
        return new EvalBlenderFile(params);
    }
    else if (descName == "PythonNode")
    {
        return new PythonNode(params);
    }
    else if (descName == "GenerateCommands")
    {
        return new CommandNode(params);
    }
    else if (descName == "PythonMaterialNode")
    {
        return new PythonMaterialNode(params);
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
        pair.second->onZoomed();
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
        m_nodes[id]->nodePosChangedSignal();
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
    if (role == ROLE_NODE_DATACHANGED)
    {
        QVariant varDataChanged = idx.data(ROLE_NODE_DATACHANGED);
        if (varDataChanged.canConvert<bool>())
        {
            bool ret = varDataChanged.toBool();
            if (m_nodes.find(id) != m_nodes.end())
                m_nodes[id]->onMarkDataChanged(ret);
        }
    }
}

void ZenoSubGraphScene::onLinkInserted(const QModelIndex& parent, int first, int last)
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex linkIdx = pGraphsModel->linkIndex(m_subgIdx, first);
    ZASSERT_EXIT(linkIdx.isValid());
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

    if (m_links.find(linkId) != m_links.end())
        return;

    ZenoNode* pInNode = m_nodes[inId];
    ZenoNode* pOutNode = m_nodes[outId];
    ZASSERT_EXIT(pInNode && pOutNode);

    ZenoFullLink* pEdge = new ZenoFullLink(QPersistentModelIndex(linkIdx), pOutNode, pInNode);
    addItem(pEdge);
    m_links[linkId] = pEdge;

    QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

    pInNode->onSocketLinkChanged(inSockIdx, true, true);
    pOutNode->onSocketLinkChanged(outSockIdx, false, true);
}

void ZenoSubGraphScene::onLinkAboutToBeRemoved(const QModelIndex&, int first, int last)
{
	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex linkIdx = pGraphsModel->linkIndex(m_subgIdx, first);
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

    QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

    if (m_nodes.find(inId) != m_nodes.end())
        m_nodes[inId]->onSocketLinkChanged(inSockIdx, true, false);
    if (m_nodes.find(outId) != m_nodes.end())
        m_nodes[outId]->onSocketLinkChanged(outSockIdx, false, false);
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

void ZenoSubGraphScene::collectNodeSelChanged(const QString& ident, bool bSelected)
{
    for (auto &pair : m_selChanges)
    {
        if (pair.first == ident)
        {
            pair.second = bSelected;
            return;
        }
    }
    m_selChanges.append({ident, bSelected});
}

void ZenoSubGraphScene::select(const QString& id)
{
    clearSelection();
    ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
    m_nodes[id]->setSelected(true);
    afterSelectionChanged();
}

void ZenoSubGraphScene::select(const QStringList& nodes)
{
    clearSelection();
    for (auto ident : nodes)
    {
        ZASSERT_EXIT(m_nodes.find(ident) != m_nodes.end());
        m_nodes[ident]->setSelected(true);
    }
    afterSelectionChanged();
}
void ZenoSubGraphScene::select(const std::vector<QString>& nodes)
{
    clearSelection();
    for (auto ident : nodes)
    {
        //ZASSERT_EXIT(m_nodes.find(ident) != m_nodes.end());
        m_nodes[ident]->setSelected(true);
    }
}
void ZenoSubGraphScene::select(const QModelIndexList &indexs) 
{
    clearSelection();
    for (auto index : indexs)
    {
        const QString &id = index.data(ROLE_OBJID).toString();
        if (m_nodes.find(id) != m_nodes.end());
            m_nodes[id]->setSelected(true);
    }
    afterSelectionChanged();
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
            QModelIndex idx = pNode->index();
            if (NO_VERSION_NODE != idx.data(ROLE_NODETYPE))
                nodesIndice.append(idx);
        }
    }
    return nodesIndice;
}

QModelIndexList ZenoSubGraphScene::selectLinkIndice() const
{
    QModelIndexList linkIndice;
    QList<QGraphicsItem*> selItems = selectedItems();
    for (auto item : selItems)
    {
        if (ZenoFullLink* pLink = qgraphicsitem_cast<ZenoFullLink*>(item))
        {
            if (pLink->isLegacyLink())
                continue;
            const QPersistentModelIndex& idx = pLink->linkInfo();
            linkIndice.append(idx);
        }
    }
    return linkIndice;
}

void ZenoSubGraphScene::copy()
{
    QList<QGraphicsItem*> selItems = this->selectedItems();
    if (selItems.isEmpty())
        return;

    //first record all nodes.
    QModelIndexList selNodes = selectNodesIndice();
    QModelIndexList selLinks = selectLinkIndice();
    QPair<NODES_DATA, LINKS_DATA> datas = UiHelper::dumpNodes(selNodes, selLinks);
    ZsgWriter::getInstance().dumpToClipboard(datas.first);
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

        QMap<QString, NODE_DATA> nodes;
        QList<EdgeInfo> links;
        QString subgName = m_subgIdx.data(ROLE_OBJNAME).toString();
        UiHelper::reAllocIdents(subgName, acceptor.nodes(), acceptor.links(), nodes, links);
        UiHelper::renameNetLabels(pGraphsModel, m_subgIdx, nodes);

        //todo: ret value for api.
        pGraphsModel->importNodes(nodes, links, pos, m_subgIdx, true);

        //mark selection for all nodes.
        clearSelection();
        for (QString ident : nodes.keys())
        {
            ZASSERT_EXIT(m_nodes.find(ident) != m_nodes.end());
            m_nodes[ident]->setSelected(true);
            collectNodeSelChanged(ident, true);
        }
        afterSelectionChanged();
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

void ZenoSubGraphScene::onSocketClicked(ZenoSocketItem* pSocketItem, Qt::MouseButton button)
{
    if (m_tempLink)
        return;

    if (button == Qt::RightButton) {
        return;
    }

    ZASSERT_EXIT(pSocketItem);

    QModelIndex paramIdx = pSocketItem->paramIndex();
    ZASSERT_EXIT(paramIdx.isValid());

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
        QPersistentModelIndex linkIdx = linkIndice[0];

        //disconnect the old link.
        const QString& outNode = linkIdx.data(ROLE_OUTNODE).toString();
        const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
        const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

        //remove current link at view.
        viewRemoveLink(linkIdx);

        socketPos = m_nodes[outNode]->getSocketPos(outSockIdx);
        ZenoSocketItem* pOutSocketItem = m_nodes[outNode]->getSocketItem(outSockIdx);
        m_tempLink = new ZenoTempLink(pOutSocketItem, outNode, socketPos, false, QModelIndexList());
        m_tempLink->setOldLink(linkIdx);
        addItem(m_tempLink);

        pSocketItem->setSockStatus(ZenoSocketItem::STATUS_TRY_DISCONN);
    }
    else
    {
        //cihou zxx: sort this nodelist by pos.y()
        QModelIndexList selNodes = selectNodesIndice();
        std::sort(selNodes.begin(), selNodes.end(), [](const QModelIndex& p1, const QModelIndex& p2) {
            int y1 = p1.data(ROLE_OBJPOS).toPointF().y();
            int y2 = p2.data(ROLE_OBJPOS).toPointF().y();
            return y1 < y2;
        });

        m_tempLink = new ZenoTempLink(pSocketItem, nodeid, socketPos, bInput, selNodes);
        addItem(m_tempLink);
        pSocketItem->setSockStatus(ZenoSocketItem::STATUS_TRY_CONN);
    }
}

void ZenoSubGraphScene::onNodePosChanged() 
{
    ZenoNode *senderNode = dynamic_cast<ZenoNode *>(sender());
    GroupNode *blackboardNode = dynamic_cast<GroupNode *>(senderNode);
    for (auto pair : m_nodes) 
    {
        ZenoNode *zenoNode = pair.second;
        if (zenoNode == senderNode) 
        {
            continue;
        }
        GroupNode *currBlackboardNode = dynamic_cast<GroupNode *>(zenoNode);
        if (blackboardNode) 
        {
            if (currBlackboardNode) 
            {
                currBlackboardNode->nodePosChanged(senderNode);
            }
            blackboardNode->nodePosChanged(zenoNode);
        } 
        else if (currBlackboardNode) 
        {
            currBlackboardNode->nodePosChanged(senderNode);
        }
    }
}

void ZenoSubGraphScene::onSocketAbsorted(const QPointF& mousePos)
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
    ZASSERT_EXIT(pGraphsModel);

    ZenoSocketItem* targetSock = m_tempLink->getAdsorbedSocket();
    if (targetSock && targetSock->isEnabled())
    {
        bool bTargetIsInput = targetSock->isInputSocket();

        QString fixedNodeId;
        bool fixedInput = false;
        QPointF fixedPos;
        m_tempLink->getFixedInfo(fixedNodeId, fixedPos, fixedInput);

        if (bTargetIsInput != fixedInput)
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
            if (bTargetIsInput && (inProp & SOCKPROP_DICTLIST_PANEL))
            {
                QString inSockType = toSockIdx.data(ROLE_PARAM_TYPE).toString();
                SOCKET_PROPERTY outProp = (SOCKET_PROPERTY)fromSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
                QString outSockType = fromSockIdx.data(ROLE_PARAM_TYPE).toString();
                QAbstractItemModel *pKeyObjModel =
                    QVariantPtr<QAbstractItemModel>::asPtr(toSockIdx.data(ROLE_VPARAM_LINK_MODEL));

                bool outSockIsContainer = false;
                if (inSockType == "list")
                {
                    outSockIsContainer = outSockType == "list";
                }
                else if (inSockType == "dict")
                {
                    const QModelIndex& fromNodeIdx = fromSockIdx.data(ROLE_NODE_IDX).toModelIndex();
                    const QString& outNodeCls = fromNodeIdx.data(ROLE_OBJNAME).toString();
                    const QString& outSockName = fromSockIdx.data(ROLE_PARAM_NAME).toString();
                    outSockIsContainer = outSockType == "dict" || (outNodeCls == "FuncBegin" && outSockName == "args");
                }

                //if outSock is a container, connects it as usual.
                if (outSockIsContainer)
                {
                    //legacy dict/list connection, and then we have to remove all inner dict key connection.
                    ZASSERT_EXIT(pKeyObjModel);
                    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
                    {
                        const QModelIndex& keyIdx = pKeyObjModel->index(r, 0);
                        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                        for (QPersistentModelIndex _linkIdx : links)
                        {
                            pGraphsModel->removeLink(_linkIdx, true);
                        }
                    }
                }
                else
                {
                    //check multiple links
                    QModelIndexList fromSockets;
                    //check selected nodes.
                    //model: ViewParamModel
                    QString paramName = fromSockIdx.data(ROLE_PARAM_NAME).toString();
                    QString paramType = fromSockIdx.data(ROLE_PARAM_TYPE).toString();

                    QModelIndexList nodes = m_tempLink->selNodes();
                    for (QModelIndex nodeIdx : nodes)
                    {
                        QString ident_ = nodeIdx.data(ROLE_OBJID).toString();
                        //model: SubGraphModel
                        OUTPUT_SOCKETS outputs = nodeIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
                        if (outputs.find(paramName) != outputs.end() &&
                            outputs[paramName].info.type == paramType)
                        {
                            OUTPUT_SOCKET outSock = outputs[paramName];
                            fromSockets.append(outputs[paramName].retIdx);
                        }
                    }
                    if (fromSockets.size() > 1)
                    {
                        QStringList toSockName = toSockIdx.data(ROLE_OBJPATH).value<QStringList>();
                        for (QModelIndex fromSockIdx : fromSockets)
                        {
                            QString ident_ = fromSockIdx.data(ROLE_OBJID).toString();
                            int n = pKeyObjModel->rowCount();
                            pGraphsModel->addExecuteCommand(new DictKeyAddRemCommand(true, pGraphsModel, toSockName, n));
                            toSockIdx = pKeyObjModel->index(n, 0);
                            pGraphsModel->addLink(m_subgIdx, fromSockIdx, toSockIdx, true);
                        }
                        return;
                    }

                    QStringList toSockName = toSockIdx.data(ROLE_OBJPATH).value<QStringList>();

                    // link to inner dict key automatically.
                    int n = pKeyObjModel->rowCount();
                    pGraphsModel->addExecuteCommand(
                        new DictKeyAddRemCommand(true, pGraphsModel, toSockIdx.data(ROLE_OBJPATH).value<QStringList>(), n));
                    toSockIdx = pKeyObjModel->index(n, 0);
                }
            }

            //remove the edge in inNode:inSock, if exists.
            if (bTargetIsInput && inProp != SOCKPROP_MULTILINK)
            {
                QPersistentModelIndex linkIdx;
                const QModelIndex& paramIdx = targetSock->paramIndex();
                const PARAM_LINKS& links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                if (!links.isEmpty())
                    linkIdx = links[0];
                if (linkIdx.isValid())
                    pGraphsModel->removeLink(linkIdx, true);
            }

            pGraphsModel->addLink(m_subgIdx, fromSockIdx, toSockIdx, true);
            return;
        }
    }

    const QPersistentModelIndex& oldLink = m_tempLink->oldLink();
    if (oldLink.isValid())
    {
        pGraphsModel->removeLink(oldLink, true);
    }

    if (!targetSock)
    {
        ZenoSocketItem* pSocketItem = m_tempLink->getFixedSocket();
        if (pSocketItem)
        {
            QModelIndex paramIdx = pSocketItem->paramIndex();
            PARAM_LINKS links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (links.isEmpty())
                pSocketItem->setSockStatus(ZenoSocketItem::STATUS_NOCONN);
        }
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

    //catch selection:
    afterSelectionChanged();
}

void ZenoSubGraphScene::afterSelectionChanged()
{
    if (!m_selChanges.empty())
    {
        ZenoMainWindow *mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        QModelIndexList selNodes, unSelNodes;
        for (auto pair : m_selChanges) {
            const QString& ident = pair.first;
            bool bSelected = pair.second;
            if (m_nodes.find(ident) == m_nodes.end())
                continue;
            ZenoNode* pNode = m_nodes[ident];
            ZASSERT_EXIT(pNode);
            if (bSelected)
                selNodes.push_back(pNode->index());
            else
                unSelNodes.push_back(pNode->index());
        }
        mainWin->onNodesSelected(m_subgIdx, unSelNodes, false);
        mainWin->onNodesSelected(m_subgIdx, selectNodesIndice(), true);
        updateKeyFrame();
    }
    m_selChanges.clear();
}

void ZenoSubGraphScene::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    //send to ZenoNode.
    QGraphicsScene::contextMenuEvent(event);
}

void ZenoSubGraphScene::focusOutEvent(QFocusEvent* event)
{
    QGraphicsScene::focusOutEvent(event);
}

void ZenoSubGraphScene::clearLayout(const QModelIndex& subGpIdx)
{
    if (subGpIdx == m_subgIdx)
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
        if (qobject_cast<GroupNode *>(pNode)) 
        {
            GroupNode *pBlackboard = qobject_cast<GroupNode *>(pNode);
            for (auto item : pBlackboard->getChildItems()) {
                GroupNode *pNewGroup = pBlackboard->getGroupNode();
                if (pNewGroup)
                    pNewGroup->appendChildItem(item);
                else
                    item->setGroupNode(pNewGroup);
            }
        }
        GroupNode *pGroup = pNode->getGroupNode();
        if (pGroup) {
            pGroup->removeChildItem(pNode);
        }
        removeItem(pNode);
        delete pNode;
        m_nodes.erase(id);
    }
    updateKeyFrame();
}

void ZenoSubGraphScene::onRowsInserted(const QModelIndex& subgIdx, const QModelIndex& parent, int first, int last)
{//right click goes here
    if (subgIdx != m_subgIdx)
        return;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex idx = pGraphsModel->index(first, m_subgIdx);
    ZenoNode *pNode = createNode(idx, m_nodeParams);
    connect(pNode, &ZenoNode::socketClicked, this, &ZenoSubGraphScene::onSocketClicked);
    connect(pNode, &ZenoNode::nodePosChangedSignal, this, &ZenoSubGraphScene::onNodePosChanged);
    pNode->initUI(this, m_subgIdx, idx);
    addItem(pNode);
    QString id = pNode->nodeId();
    m_nodes[id] = pNode;

    if (dynamic_cast<GroupNode *>(pNode)) 
    {
        QRectF rect;
        for (auto item : selectedItems()) 
        {
            if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item)) {
                rect = rect.united(QRectF(item->scenePos(), item->boundingRect().size()));
            }
        }
        if (rect.isValid()) 
        {
            int width = ZenoStyle::dpiScaled(50);
            rect.adjust(-width, -width, width, width);
            GroupNode *pGroup = dynamic_cast<GroupNode *>(pNode);
            pGroup->resize(rect.size());
            pGroup->updateBlackboard();
            pGroup->updateNodePos(rect.topLeft(), false);
            for (auto item : selectedItems()) 
            {
                ZenoNode *pChildNode = dynamic_cast<ZenoNode *>(item);
                if (pChildNode)
                    pGroup->appendChildItem(pChildNode);
            }
        }
    }
    if (m_subgIdx.data(ROLE_OBJNAME).toString() == "main") {  //new nodes mark dirty
        pGraphsModel->markNodeDataChanged(idx, false);
        pNode->onMarkDataChanged(true);
    }
}

void ZenoSubGraphScene::selectObjViaNodes() {
    // FIXME temp function for merge
    // for selecting objects in viewport via selected nodes
    ZenoMainWindow* pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);
    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    for (auto pDisplay : views) {
        ZASSERT_EXIT(pDisplay);
        auto pZenovis = pDisplay->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto scene = pZenovis->getSession()->get_scene();
        ZASSERT_EXIT(scene);

        QList<QGraphicsItem*> selItems = this->selectedItems();
        auto picker = pDisplay->picker();
        ZASSERT_EXIT(picker);
        picker->clear();
        for (auto item : selItems) {
            if (auto *pNode = qgraphicsitem_cast<ZenoNode *>(item)) {
                auto node_id = pNode->index().data(ROLE_OBJID).toString().toStdString();
                auto key = zeno::getSession().globalComm->getObjKeyByObjID(node_id);
                if (key != "")
                    picker->add(key);
            }
        }
        picker->sync_to_scene();
        zenoApp->getMainWindow()->updateViewport();
    }
}

void ZenoSubGraphScene::updateKeyFrame() 
{
    QVector<int> keys;
    for (const QModelIndex &index : selectNodesIndice()) {
        keys << index.data(ROLE_KEYFRAMES).value<QVector<int>>();
    }
    qSort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    zenoApp->getMainWindow()->timeline()->updateKeyFrames(keys);
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
            QList<QPersistentModelIndex> links, legacylinks;
            QList<QPersistentModelIndex> netLabels;
            for (auto item : selItems)
            {
                if (ZenoNode* pNode = qgraphicsitem_cast<ZenoNode*>(item))
                {
                    nodes.append(pNode->index());
                }
                else if (ZenoFullLink* pLink = qgraphicsitem_cast<ZenoFullLink*>(item))
                {
                    if (pLink->isLegacyLink())
                    {
                        legacylinks.append(pLink->linkInfo());
                        removeItem(pLink);
                    }
                    else
                        links.append(pLink->linkInfo());
                }
                else if (ZGraphicsNetLabel* pNetLabel = qgraphicsitem_cast<ZGraphicsNetLabel*>(item))
                {
                    netLabels.append(pNetLabel->paramIdx());
                }
            }

            IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
            ZASSERT_EXIT(pGraphsModel);
            if (!nodes.isEmpty() || !links.isEmpty() || !netLabels.isEmpty())
            {
                pGraphsModel->beginTransaction("remove nodes and links");
                zeno::scope_exit scope([=]() { pGraphsModel->endTransaction(); });

                for (QPersistentModelIndex linkIdx : links)
                {
                    pGraphsModel->removeLink(linkIdx, true);
                }
                for (QPersistentModelIndex nodeIdx : nodes)
                {
                    QString id = nodeIdx.data(ROLE_OBJID).toString();
                    pGraphsModel->removeNode(id, m_subgIdx, true);
                }
                if (nodes.isEmpty() && !netLabels.isEmpty())
                {
                    for (const QPersistentModelIndex& socketIdx : netLabels)
                    {
                        pGraphsModel->removeNetLabel(m_subgIdx, socketIdx);
                    }
                }
            }
            for (auto linkIdx : legacylinks)
            {
                pGraphsModel->removeLegacyLink(linkIdx);
            }
        }
    }
    else if (!event->isAccepted() && (event->modifiers() & Qt::ControlModifier) && event->key() == Qt::Key_G) {
        // FIXME temp function for merge
        selectObjViaNodes();
    }
    int uKey = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    if (modifiers & Qt::ShiftModifier) {
        uKey += Qt::SHIFT;
    }
    if (modifiers & Qt::ControlModifier) {
        uKey += Qt::CTRL;
    }
    if (modifiers & Qt::AltModifier) {
        uKey += Qt::ALT;
    }
    if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_SelectAllNodes)) 
    {
        for (const auto& it : m_nodes) {
            it.second->setSelected(true);
        }
        for (const auto& it : m_links) {
            it->setSelected(true);
        }
    } 
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Once)) 
    {
        updateNodeStatus(m_bOnceOn, OPT_ONCE);
    } 
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Bypass))
    {
        updateNodeStatus(m_bBypassOn, OPT_MUTE);
    } 
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_View)) 
    {
        int options = OPT_VIEW;
        updateNodeStatus(m_bViewOn, options);
    }
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Cache))
    {
        int options = OPT_CACHE;
        if (m_bCacheOn) {
            options |= OPT_VIEW;
            m_bViewOn = false;
        }
        updateNodeStatus(m_bCacheOn, options);
    }
}

void ZenoSubGraphScene::updateNodeStatus(bool &bOn, int option) 
{
    bOn = !bOn;
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    for (const QModelIndex &idx : selectNodesIndice()) 
    {
        STATUS_UPDATE_INFO info;
        int options = idx.data(ROLE_OPTIONS).toInt();
        info.oldValue = options;
        if (bOn)
            options |= option;
        else
            options &= (~option);
        info.role = ROLE_OPTIONS;
        info.newValue = options;
        pGraphsModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, m_subgIdx);
    }
    for (const QModelIndex& idx : selectNodesIndice())
    {
        STATUS_UPDATE_INFO info;
        int options = idx.data(ROLE_OPTIONS).toInt();
        info.oldValue = options;
        if (options & OPT_VIEW)
            options |= OPT_CACHE;
        else
            continue;
        info.role = ROLE_OPTIONS;
        info.newValue = options;
        pGraphsModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, m_subgIdx);
    }
    
}
void ZenoSubGraphScene::keyReleaseEvent(QKeyEvent* event)
{
    QGraphicsScene::keyReleaseEvent(event);
}
