#include "zenosubgraphscene.h"
#include "zenonode.h"
#include "subnetnode.h"
#include "heatmapnode.h"
#include "cameranode.h"
#include "pythonnode.h"
#include "zenolink.h"
#include <zeno/io/zsg2reader.h>
#include <zeno/io/zenwriter.h>
#include "util/uihelper.h"
#include "uicommon.h"
#include "nodeeditor/gv/nodegrid.h"
#include "model/GraphModel.h"
#include "model/LinkModel.h"
#include "model/parammodel.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include <zeno/utils/log.h>
#include "util/log.h"
#include "variantptr.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "nodeeditor/gv/groupnode.h"
#include "style/zenostyle.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "zenomainwindow.h"
#include <zenovis/ObjectsManager.h>
#include "viewport/picker.h"
#include "settings/zenosettingsmanager.h"
#include "widgets/ztimeline.h"
#include "zenosubgraphview.h"
//#include "nodeeditor/gv/pythonmaterialnode.h"

ZForegroundItem::ZForegroundItem(QGraphicsItem* parent)
    : QGraphicsRectItem(parent)
{
    setZValue(ZVALUE_POPUPWIDGET);
}

ZForegroundItem::~ZForegroundItem()
{
}

QRectF ZForegroundItem::boundingRect() const
{
    if (!scene()->views().isEmpty())
    {
        const QGraphicsView* pView = this->scene()->views().first();
        return pView->mapToScene(pView->viewport()->rect()).boundingRect();
    }
    return QRectF();
}

void ZForegroundItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);
    GraphModel* pModel = pScene->getGraphModel();
    if (pModel && pModel->isLocked())
    {
        painter->setOpacity(0.3);
        painter->setBrush(QColor(83, 83, 85));
        const auto& rect = boundingRect();
        painter->fillRect(rect, QColor(83, 83, 85));
    }
}

ZenoSubGraphScene::ZenoSubGraphScene(QObject *parent)
    : QGraphicsScene(parent)
    , m_tempLink(nullptr)
    , m_bOnceOn(false)
    , m_bBypassOn(false)
    , m_bViewOn(false)
    , m_model(nullptr)
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

GraphModel* ZenoSubGraphScene::getGraphModel() const
{
    return m_model;
}

void ZenoSubGraphScene::initModel(GraphModel* pGraphM)
{
    m_model = pGraphM;

    ZForegroundItem* foreItem = new ZForegroundItem;
    addItem(foreItem);

    //disconnect(m_model, SIGNAL(reloaded()), this, SLOT(reload()));
    //disconnect(m_model, SIGNAL(clearLayout()), this, SLOT(clearLayout()));
    disconnect(m_model, &GraphModel::dataChanged, this, &ZenoSubGraphScene::onDataChanged);
    disconnect(m_model, &GraphModel::rowsAboutToBeRemoved, this, &ZenoSubGraphScene::onRowsAboutToBeRemoved);
	disconnect(m_model, &GraphModel::rowsInserted, this, &ZenoSubGraphScene::onRowsInserted);

    QVector<ZenoNode *> blackboardVect;
    for (int r = 0; r < pGraphM->rowCount(); r++)
    {
        const QModelIndex& idx = pGraphM->index(r, 0);
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
    }

    for (auto it : m_nodes)
    {
        ZenoNode *inNode = it.second;
        const QString& id = inNode->nodeId();
        const QModelIndex& idx = m_model->indexFromName(id);

        ParamsModel* viewParams = QVariantPtr<ParamsModel>::asPtr(idx.data(ROLE_PARAMS));
        for (int r = 0; r < viewParams->rowCount(); r++)
        {
            const QModelIndex& paramIdx = viewParams->index(r, 0);
            if (!paramIdx.data(ROLE_ISINPUT).toBool())
                continue;

            const QString& inSock = paramIdx.data(ROLE_PARAM_NAME).toString();
            const int inSockProp = paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();

            PARAM_LINKS links = paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
            {
                for (const QPersistentModelIndex& linkIdx : links)
                {
                    initLink(linkIdx);
                }
            }
            else
            {
                //TODO: refactor dict/list case
#if 0
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
#endif
            }
        }

#if 0
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
#endif
    }

    //TODO:
#if 0
    LinkModel* pLegacyLinks = pGraphsModel->legacyLinks(m_subgIdx);
    if (pLegacyLinks)
    {
        for (int r = 0; r < pLegacyLinks->rowCount(); r++)
        {
            QModelIndex linkIdx = pLegacyLinks->index(r, 0);
            initLink(linkIdx);
        }
    }
#endif
    //connect(m_model, SIGNAL(reloaded()), this, SLOT(reload()));
    //connect(m_model, SIGNAL(clearLayout()), this, SLOT(clearLayout()));
    connect(m_model, &GraphModel::dataChanged, this, &ZenoSubGraphScene::onDataChanged);
    connect(m_model, &GraphModel::rowsAboutToBeRemoved, this, &ZenoSubGraphScene::onRowsAboutToBeRemoved);
    connect(m_model, &GraphModel::rowsInserted, this, &ZenoSubGraphScene::onRowsInserted);
    connect(m_model, &GraphModel::nameUpdated, this, &ZenoSubGraphScene::onNameUpdated);

    //link sync:
    QAbstractItemModel* pLinkModel = m_model->getLinkModel();
    connect(pLinkModel, &QAbstractItemModel::rowsInserted, this, &ZenoSubGraphScene::onLinkInserted);
    connect(pLinkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &ZenoSubGraphScene::onLinkAboutToBeRemoved);
}

void ZenoSubGraphScene::initLink(const QModelIndex& linkIdx)
{
    if (!linkIdx.isValid())
        return;

    zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    QUuid linkid = linkIdx.data(ROLE_LINKID).toUuid();
    if (m_links.find(linkid) != m_links.end())
        return;

    const QString& inId = QString::fromStdString(edge.inNode);
    const QString& outId = QString::fromStdString(edge.outNode);
    const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    const QModelIndex& inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    const QString inKey = QString::fromStdString(edge.inKey);
    const QString outKey = QString::fromStdString(edge.outKey);

    ZenoNode* inNode = m_nodes[inId];
    ZenoNode* outNode = m_nodes[outId];
    ZASSERT_EXIT(inNode && outNode);

    ZenoFullLink *pEdge = new ZenoFullLink(linkIdx, outNode, inNode);
    addItem(pEdge);
    m_links[linkid] = pEdge;

    ZenoSocketItem *socketItem = outNode->getSocketItem(outSockIdx, outKey);
    ZASSERT_EXIT(socketItem);
    socketItem->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
    //socketItem->toggle(true);

    socketItem = inNode->getSocketItem(inSockIdx, inKey);
    ZASSERT_EXIT(socketItem);
    socketItem->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
    //socketItem->toggle(true);
}

ZenoNode* ZenoSubGraphScene::createNode(const QModelIndex& idx, const NodeUtilParam& params)
{
    const QString& descName = idx.data(ROLE_CLASS_NAME).toString();
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
    else if (descName == "PythonNode")
    {
        return new PythonNode(params);
    }
    //else if (descName == "PythonMaterialNode")
    //{     //TODO:
    //    return new PythonMaterialNode(params);
    //}
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

void ZenoSubGraphScene::onNameUpdated(const QModelIndex& nodeIdx, const QString& oldName)
{
    const QString& newName = nodeIdx.data(ROLE_NODE_NAME).toString();
    ZASSERT_EXIT(newName != oldName);
    m_nodes[newName] = m_nodes[oldName];
    m_nodes.erase(oldName);
    m_nodes[newName]->onNameUpdated(newName);
}

void ZenoSubGraphScene::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    QModelIndex idx = topLeft;
    int role = roles[0];

    QString id = idx.data(ROLE_NODE_NAME).toString();

    if (role == ROLE_OBJPOS)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        QVariant var = idx.data(ROLE_OBJPOS);
        if (var.type() == QVariant::List) {
            QVariantList lst = var.toList();
            ZASSERT_EXIT(lst.size() == 2);
            QPointF pos(lst[0].toFloat(), lst[1].toFloat());
            m_nodes[id]->setPos(pos);
            m_nodes[id]->nodePosChangedSignal();
        }
        else if (var.type() == QVariant::PointF) {
            QPointF pos = idx.data(ROLE_OBJPOS).toPoint();
            m_nodes[id]->setPos(pos);
            m_nodes[id]->nodePosChangedSignal();
        }
    }
    if (role == ROLE_NODE_STATUS)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        int options = idx.data(ROLE_NODE_STATUS).toInt();
        m_nodes[id]->onOptionsUpdated(options);
    }
    if (role == ROLE_NODE_ISVIEW)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());

    }
    if (role == ROLE_COLLASPED)
    {
        ZASSERT_EXIT(m_nodes.find(id) != m_nodes.end());
        bool bCollasped = idx.data(ROLE_COLLASPED).toBool();
        m_nodes[id]->onCollaspeUpdated(bCollasped);
    }
    if (role == ROLE_NODE_DIRTY)
    {
        QVariant varDataChanged = idx.data(ROLE_NODE_DIRTY);
        if (varDataChanged.canConvert<bool>())
        {
            bool bDirty = varDataChanged.toBool();
            if (m_nodes.find(id) != m_nodes.end())
                m_nodes[id]->onMarkDataChanged(bDirty);
        }
    }
#if 0
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
#endif
}

void ZenoSubGraphScene::onLinkInserted(const QModelIndex& parent, int first, int last)
{
    LinkModel* pModel = qobject_cast<LinkModel*>(sender());
    ZASSERT_EXIT(pModel);
    QModelIndex linkIdx = pModel->index(first, 0, parent);
    ZASSERT_EXIT(linkIdx.isValid());
    viewAddLink(linkIdx);
}

void ZenoSubGraphScene::viewAddLink(const QModelIndex& linkIdx)
{
    zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    QUuid linkid = linkIdx.data(ROLE_LINKID).toUuid();

    const QString& inId = QString::fromStdString(edge.inNode);
    const QString& inSock = QString::fromStdString(edge.inParam);
    const QString inKey = QString::fromStdString(edge.inKey);
    const QString& outId = QString::fromStdString(edge.outNode);
    const QString& outSock = QString::fromStdString(edge.outParam);
    const QString outKey = QString::fromStdString(edge.outKey);

    if (m_nodes.find(inId) == m_nodes.end() || m_nodes.find(outId) == m_nodes.end())
    {
        //todo: half link across from two different subgraph.
        return;
    }

    if (m_links.find(linkid) != m_links.end())
        return;

    ZenoNode* pInNode = m_nodes[inId];
    ZenoNode* pOutNode = m_nodes[outId];
    ZASSERT_EXIT(pInNode && pOutNode);

    ZenoFullLink* pEdge = new ZenoFullLink(QPersistentModelIndex(linkIdx), pOutNode, pInNode);
    addItem(pEdge);
    m_links[linkid] = pEdge;

    QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

    pInNode->onSocketLinkChanged(inSockIdx, true, true, inKey);
    pOutNode->onSocketLinkChanged(outSockIdx, false, true, outKey);
}

void ZenoSubGraphScene::onLinkAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    LinkModel* pModel = qobject_cast<LinkModel*>(sender());
    ZASSERT_EXIT(pModel);
    QModelIndex linkIdx = pModel->index(first, 0, parent);
    ZASSERT_EXIT(linkIdx.isValid());
    viewRemoveLink(linkIdx);
}

void ZenoSubGraphScene::viewRemoveLink(const QModelIndex& linkIdx)
{
    zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();
    QUuid linkid = linkIdx.data(ROLE_LINKID).toUuid();
    if (m_links.find(linkid) == m_links.end())
        return;

    ZenoFullLink* pLink = m_links[linkid];
    m_links.remove(linkid);
    delete pLink;

    const QString& inId = QString::fromStdString(edge.inNode);
    const QString& inSock = QString::fromStdString(edge.inParam);
    const QString inKey = QString::fromStdString(edge.inKey);
    const QString& outId = QString::fromStdString(edge.outNode);
    const QString& outSock = QString::fromStdString(edge.outParam);
    const QString outKey = QString::fromStdString(edge.outKey);

    QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

    if (m_nodes.find(inId) != m_nodes.end())
        m_nodes[inId]->onSocketLinkChanged(inSockIdx, true, false, inKey);
    if (m_nodes.find(outId) != m_nodes.end())
        m_nodes[outId]->onSocketLinkChanged(outSockIdx, false, false, outKey);
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

void ZenoSubGraphScene::collectNodeSelChanged(const QString& name, bool bSelected)
{
    for (auto &pair : m_selChanges)
    {
        if (pair.first == name)
        {
            pair.second = bSelected;
            return;
        }
    }
    m_selChanges.append({name, bSelected});
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
    for (auto name : nodes)
    {
        ZASSERT_EXIT(m_nodes.find(name) != m_nodes.end());
        m_nodes[name]->setSelected(true);
    }
    afterSelectionChanged();
}
void ZenoSubGraphScene::select(const std::vector<QString>& nodes)
{
    clearSelection();
    for (auto name : nodes)
    {
        //ZASSERT_EXIT(m_nodes.find(name) != m_nodes.end());
        m_nodes[name]->setSelected(true);
    }
}
void ZenoSubGraphScene::select(const QModelIndexList &indexs) 
{
    clearSelection();
    for (auto index : indexs)
    {
        const QString &id = index.data(ROLE_NODE_NAME).toString();
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
    for (QString name : m_errNodes)
    {
        if (m_nodes.find(name) != m_nodes.end())
        {
            m_nodes[name]->markError(false);
        }
    }
    m_errNodes.clear();
}

void ZenoSubGraphScene::undo()
{
    ZASSERT_EXIT(m_model);
    m_model->undo();
}

void ZenoSubGraphScene::redo()
{
    ZASSERT_EXIT(m_model);
    m_model->redo();
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
            if (zeno::NoVersionNode != idx.data(ROLE_NODETYPE))
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

void ZenoSubGraphScene::save()
{
    ZASSERT_EXIT(m_model);
    QStringList path = m_model->currentPath();
    ZASSERT_EXIT(!path.empty());
    const QString& projName = path[0];
    UiHelper::saveProject(projName);
}

void ZenoSubGraphScene::copy()
{
    QList<QGraphicsItem*> selItems = this->selectedItems();
    if (selItems.isEmpty())
        return;

    //first record all nodes.
    QModelIndexList selNodes = selectNodesIndice();
    QModelIndexList selLinks = selectLinkIndice();
    QPair<zeno::NodesData, zeno::LinksData> datas = UiHelper::dumpNodes(selNodes, selLinks);
    //ZsgWriter::getInstance().dumpToClipboard(datas.first);
}

void ZenoSubGraphScene::paste(QPointF pos)
{
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    //TODO: paste io
#if 0
    IGraphsModel *pGraphsModel = zenoApp->graphsManager()->currentModel();
    if (pMimeData->hasText() && pGraphsModel)
    {
        const QString& strJson = pMimeData->text();

        TransferAcceptor acceptor(pGraphsModel);
        Zsg2Reader::getInstance().importNodes(pGraphsModel, m_subgIdx, strJson, pos, &acceptor);

        QMap<QString, NODE_DATA> nodes;
        QList<EdgeInfo> links;
        QString subgName = m_subgIdx.data(ROLE_CLASS_NAME).toString();
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
#endif
}

void ZenoSubGraphScene::reload(const QModelIndex& subGpIdx)
{
    clear();
    GraphModel* pGraphM = qobject_cast<GraphModel*>(sender());
    initModel(pGraphM);
}

void ZenoSubGraphScene::onSocketClicked(ZenoSocketItem* pSocketItem, zeno::LinkFunction lnkProp)
{
    if (m_tempLink)
        return;

    ZASSERT_EXIT(pSocketItem);

    QModelIndex paramIdx = pSocketItem->paramIndex();
    ZASSERT_EXIT(paramIdx.isValid());

    bool bInput = pSocketItem->isInputSocket();
    QString nodeid = pSocketItem->nodeIdent();

    zeno::ParamControl ctrl = (zeno::ParamControl)paramIdx.data(ROLE_PARAM_CONTROL).toInt();
    SOCKET_PROPERTY prop = (SOCKET_PROPERTY)paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();
    QPointF socketPos = pSocketItem->center();

    ZASSERT_EXIT(m_nodes.find(nodeid) != m_nodes.end());

    PARAM_LINKS linkIndice = paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    bool bDisconnetLink = prop != SOCKPROP_MULTILINK && bInput && !linkIndice.isEmpty();
    //it's difficult to handle the situation when trying to disconnect the link.
    if (false && bDisconnetLink)
    {
        QPersistentModelIndex linkIdx = linkIndice[0];

        //disconnect the old link.
        zeno::EdgeInfo edge = linkIdx.data(ROLE_LINK_INFO).value<zeno::EdgeInfo>();

        const QString& outNode = QString::fromStdString(edge.outNode);
        const QString& outSock = QString::fromStdString(edge.outParam);
        const QString outKey = QString::fromStdString(edge.outKey);
        const QModelIndex& outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();

        //remove current link at view.
        viewRemoveLink(linkIdx);

        socketPos = m_nodes[outNode]->getSocketPos(outSockIdx);
        ZenoSocketItem* pOutSocketItem = m_nodes[outNode]->getSocketItem(outSockIdx, outKey);
        m_tempLink = new ZenoTempLink(pOutSocketItem, outNode, socketPos, false, lnkProp, QModelIndexList());
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

        m_tempLink = new ZenoTempLink(pSocketItem, nodeid, socketPos, bInput, lnkProp, selNodes);
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
    zeno::LinkFunction lnkProp = zeno::Link_Copy;
    m_tempLink->getFixedInfo(nodeId, fixedPos, bFixedInput, lnkProp);

    QPointF pos = mousePos;
    QList<QGraphicsItem *> catchedItems = items(pos);
    QList<ZenoNode *> catchNodes;
    QList<ZenoSocketItem* > catchSocks;
    for (QGraphicsItem *item : catchedItems)
    {
        if (ZenoNode *pNode = qgraphicsitem_cast<ZenoNode *>(item))
        {
            if (pNode->index().data(ROLE_NODE_NAME).toString() != nodeId)
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

    ZenoSocketItem* targetSock = m_tempLink->getAdsorbedSocket();
    if (targetSock && targetSock->isEnabled())
    {
        bool bTargetIsInput = targetSock->isInputSocket();

        QString fixedNodeId;
        bool fixedInput = false;
        QPointF fixedPos;
        zeno::LinkFunction lnkProp = zeno::Link_Copy;
        m_tempLink->getFixedInfo(fixedNodeId, fixedPos, fixedInput, lnkProp);

        if (bTargetIsInput != fixedInput)
        {
            QPersistentModelIndex outSockIdx, inSockIdx;
            if (fixedInput) {
                outSockIdx = targetSock->paramIndex();
                inSockIdx = m_tempLink->getFixedSocket()->paramIndex();
            } else {
                outSockIdx = m_tempLink->getFixedSocket()->paramIndex();
                inSockIdx = targetSock->paramIndex();
            }

            const QPersistentModelIndex& oldLink = m_tempLink->oldLink();
            if (oldLink.isValid())
            {
                //same link?
                if (oldLink.data(ROLE_OUTSOCK_IDX).toModelIndex() == outSockIdx &&
                    oldLink.data(ROLE_INSOCK_IDX).toModelIndex() == inSockIdx)
                {
                    viewAddLink(oldLink);
                    return;
                }
            }

            m_model->beginTransaction(tr("add Link"));
            zeno::scope_exit sp([=]() { m_model->endTransaction(); });

            //dict panel.
            SOCKET_PROPERTY inProp = (SOCKET_PROPERTY)inSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();
            //TODO: dict case
#if 0
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
                    const QString& outNodeCls = fromNodeIdx.data(ROLE_CLASS_NAME).toString();
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
                        QString ident_ = nodeIdx.data(ROLE_NODE_NAME).toString();
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
                        QString toSockName = toSockIdx.data(ROLE_OBJPATH).toString();
                        for (QModelIndex fromSockIdx : fromSockets)
                        {
                            QString ident_ = fromSockIdx.data(ROLE_NODE_NAME).toString();
                            int n = pKeyObjModel->rowCount();
                            pGraphsModel->addExecuteCommand(new DictKeyAddRemCommand(true, pGraphsModel, toSockName, n));
                            toSockIdx = pKeyObjModel->index(n, 0);
                            pGraphsModel->addLink(m_subgIdx, fromSockIdx, toSockIdx, true);
                        }
                        return;
                    }

                    QString toSockName = toSockIdx.data(ROLE_OBJPATH).toString();

                    // link to inner dict key automatically.
                    int n = pKeyObjModel->rowCount();
                    pGraphsModel->addExecuteCommand(
                        new DictKeyAddRemCommand(true, pGraphsModel, toSockIdx.data(ROLE_OBJPATH).toString(), n));
                    toSockIdx = pKeyObjModel->index(n, 0);
                }
            }
#endif
            QModelIndex outNodeIdx = outSockIdx.data(ROLE_NODE_IDX).toModelIndex();
            QModelIndex inNodeIdx = inSockIdx.data(ROLE_NODE_IDX).toModelIndex();

            zeno::EdgeInfo newEdge;
            newEdge.outNode = outNodeIdx.data(ROLE_NODE_NAME).toString().toStdString();
            newEdge.outParam = outSockIdx.data(ROLE_PARAM_NAME).toString().toStdString();
            newEdge.inNode = inNodeIdx.data(ROLE_NODE_NAME).toString().toStdString();
            newEdge.inParam = inSockIdx.data(ROLE_PARAM_NAME).toString().toStdString();
            newEdge.lnkfunc = lnkProp;

            if (!fixedInput)
            {
                if (zeno::Param_Dict == outSockIdx.data(ROLE_PARAM_TYPE))
                {
                    ZenoSocketItem* pOutSocket = m_tempLink->getFixedSocket();
                    const QString& outKey = pOutSocket->innerKey();
                    if (!outKey.isEmpty()) {
                        newEdge.outKey = outKey.toStdString();
                    }
                }
            }

            m_model->addLink(newEdge);
            return;
        }
    }

    const QPersistentModelIndex& oldLink = m_tempLink->oldLink();
    if (oldLink.isValid())
    {
        m_model->removeLink(oldLink);
    }

    if (!targetSock)
    {
        ZenoSocketItem* pSocketItem = m_tempLink->getFixedSocket();
        if (pSocketItem)
        {
            QModelIndex paramIdx = pSocketItem->paramIndex();
            PARAM_LINKS links = paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
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
    if (m_tempLink && !m_model->isLocked())
    {
        onSocketAbsorted(event->scenePos());
        return;
    }
    QGraphicsScene::mouseMoveEvent(event);
}

void ZenoSubGraphScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_tempLink && event->button() != Qt::MidButton && event->button() != Qt::RightButton && !m_model->isLocked())
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
            const QString& name = pair.first;
            bool bSelected = pair.second;
            if (m_nodes.find(name) == m_nodes.end())
                continue;
            ZenoNode* pNode = m_nodes[name];
            ZASSERT_EXIT(pNode);
            if (bSelected)
            {
                selNodes.push_back(pNode->index());
                pNode->setZValue(ZVALUE_SELECTED);
            }
            else
            {
                unSelNodes.push_back(pNode->index());
                pNode->setZValue(-2);
            }
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

void ZenoSubGraphScene::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    for (int r = first; r <= last; r++)
    {
        QModelIndex idx = m_model->index(r, 0, parent);
        QString id = idx.data(ROLE_NODE_NAME).toString();
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

void ZenoSubGraphScene::onRowsInserted(const QModelIndex& parent, int first, int last)
{
    //right click goes here
    QModelIndex idx = m_model->index(first, 0, parent);
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
                auto node_id = pNode->index().data(ROLE_NODE_NAME).toString().toStdString();
                for (const auto &[prim_name, _] : scene->objectsMan->pairsShared()) {
                    if (prim_name.find(node_id) != std::string::npos)
                        picker->add(prim_name);
                }
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
    if (m_model->isLocked())
        return;
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
                    //netLabels.append(pNetLabel->paramIdx());
                }
            }

            if (!nodes.isEmpty() || !links.isEmpty() || !netLabels.isEmpty())
            {
                m_model->beginTransaction("remove nodes and links");
                zeno::scope_exit scope([=]() { m_model->endTransaction(); });

                for (QPersistentModelIndex linkIdx : links)
                {
                    m_model->removeLink(linkIdx);
                }
                for (QPersistentModelIndex nodeIdx : nodes)
                {
                    QString id = nodeIdx.data(ROLE_NODE_NAME).toString();
                    QString cls = nodeIdx.data(ROLE_CLASS_NAME).toString();
                    if ("SubInput" == cls || "SubOutput" == cls) {
                        //SubInput/Output will not allow to be removed by user, 
                        //which shoule be done by edit param dialog or core api `removeSubnetArgs`.
                        zeno::log_warn("SubInput/Output will not allow to be removed by user");
                    }
                    else {
                        m_model->removeNode(id);
                    }
                }
                if (nodes.isEmpty() && !netLabels.isEmpty())
                {
                    for (const QPersistentModelIndex& socketIdx : netLabels)
                    {
                        //m_model->removeNetLabel(m_subgIdx, socketIdx);
                    }
                }
            }
            for (auto linkIdx : legacylinks)
            {
                //pGraphsModel->removeLegacyLink(linkIdx);
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
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_Bypass))
    {
        updateNodeStatus(m_bBypassOn, zeno::Mute);
    } 
    else if (!event->isAccepted() && uKey == ZenoSettingsManager::GetInstance().getShortCut(ShortCut_View)) 
    {
        updateNodeStatus(m_bViewOn, zeno::View);
    }
}

void ZenoSubGraphScene::updateNodeStatus(bool &bOn, int option) 
{
    bOn = !bOn;
    for (const QModelIndex &idx : selectNodesIndice()) 
    {
        int options = idx.data(ROLE_NODE_STATUS).toInt();
        UiHelper::qIndexSetData(idx, options, ROLE_NODE_STATUS);
    }
    
}
void ZenoSubGraphScene::keyReleaseEvent(QKeyEvent* event)
{
    QGraphicsScene::keyReleaseEvent(event);
}
