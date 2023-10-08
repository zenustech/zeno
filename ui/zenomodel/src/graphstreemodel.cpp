#include "graphstreemodel.h"
#include "graphstreemodel_impl.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include "zassert.h"
#include "apilevelscope.h"
#include "graphsmanagment.h"


GraphsTreeModel::GraphsTreeModel(QObject* parent)
    : IGraphsModel(parent)
    , m_impl(new GraphsTreeModel_impl(this))
    , m_stack(new QUndoStack(this))
    , m_apiLevel(0)
    , m_dirty(false)
    , m_bIOProcessing(false)
    , m_bApiEnableRun(true)
    , m_version(zenoio::VER_2_5)
    , m_pSubgraphs(nullptr)
{
    connect(m_impl, &QAbstractItemModel::rowsAboutToBeRemoved, this,
        [=](const QModelIndex &parent, int first, int last) {
            emit _rowsAboutToBeRemoved(parent, parent, first, last);
    });
    connect(m_impl, &QAbstractItemModel::rowsInserted, this,
        [=](const QModelIndex &parent, int first, int last) {
            emit _rowsInserted(parent, parent, first, last);
    });
    connect(m_impl, &QAbstractItemModel::dataChanged, this,
        [=](const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles) {
            ZASSERT_EXIT(!roles.isEmpty());
            emit _dataChanged(topLeft.parent(), topLeft, roles[0]);
    });
}

GraphsTreeModel::~GraphsTreeModel()
{
}

void GraphsTreeModel::initSubgraphs(IGraphsModel* pSubgraphs)
{
    m_pSubgraphs = pSubgraphs;
}

QUndoStack* GraphsTreeModel::stack() const
{
    return m_stack;
}

QModelIndex GraphsTreeModel::index(int row, int column, const QModelIndex& parent) const
{
    return m_impl->index(row, column, parent);
}

QModelIndex GraphsTreeModel::index(const QString& subGraphName) const
{
    if (subGraphName == "main")
    {
        //legacy case
        return m_impl->mainIndex();
    }
    //shared subgraph has been moved out of this model impl, except main.
    return QModelIndex();
}

QModelIndex GraphsTreeModel::index(const QString &id, const QModelIndex &subGpIdx)
{
    return m_impl->index(id, subGpIdx);
}

QModelIndex GraphsTreeModel::index(int r, const QModelIndex &subGpIdx)
{
    return m_impl->index(r, subGpIdx);
}

QModelIndex GraphsTreeModel::mainIndex() const
{
    return m_impl->mainIndex();
}

QModelIndex GraphsTreeModel::nodeIndex(const QString &ident)
{
    return m_impl->index(ident);
}

/* end: node index: */

QModelIndex GraphsTreeModel::nodeIndex(uint32_t sid, uint32_t nodeid)
{
    return m_impl->index(sid, nodeid);
}

QModelIndex GraphsTreeModel::subgIndex(uint32_t sid)
{
    return m_impl->subgIndex(sid);
}

QModelIndex GraphsTreeModel::parent(const QModelIndex &child) const
{
    return m_impl->parent(child);
}

int GraphsTreeModel::rowCount(const QModelIndex& parent) const
{
    return m_impl->rowCount(parent);
}

int GraphsTreeModel::columnCount(const QModelIndex&) const
{
    return 1;
}

QVariant GraphsTreeModel::data(const QModelIndex& index, int role) const
{
    return m_impl->data(index, role);
}

bool GraphsTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return m_impl->setData(index, value, role);
}

int GraphsTreeModel::itemCount(const QModelIndex &subGpIdx) const
{
    return m_impl->itemCount(subGpIdx);
}

QModelIndex GraphsTreeModel::linkIndex(const QModelIndex &subgIdx, int r)
{
    return m_impl->linkIndex(subgIdx, r);
}

QModelIndex GraphsTreeModel::linkIndex(
                        const QModelIndex &subgIdx,
                        const QString &outNode,
                        const QString &outSock,
                        const QString &inNode,
                        const QString &inSock)
{
    return m_impl->linkIndex(subgIdx, outNode, outSock, inNode, inSock);
}

void GraphsTreeModel::addNode(const NODE_DATA &nodeData, const QModelIndex &subGpIdx, bool enableTransaction)
{
    m_impl->addNode(nodeData, subGpIdx, enableTransaction);
}

void GraphsTreeModel::setNodeData(
                        const QModelIndex& nodeIndex,
                        const QModelIndex& subGpIdx,
                        const QVariant& value,
                        int role)
{
    m_impl->setNodeData(nodeIndex, subGpIdx, value, role);
}

void GraphsTreeModel::importNodes(
                        const QMap<QString, NODE_DATA>& nodes,
                        const QList<EdgeInfo>& links,
                        const QPointF& pos,
                        const QModelIndex& subGpIdx,
                        bool enableTransaction)
{
    m_impl->importNodes(nodes, links, pos, subGpIdx, enableTransaction);
}

void GraphsTreeModel::removeNode(const QString &nodeid, const QModelIndex &subGpIdx, bool enableTransaction)
{
    m_impl->removeNode(nodeid, subGpIdx, enableTransaction);
}


QModelIndex GraphsTreeModel::addLink(
                        const QModelIndex &subgIdx,
                        const QModelIndex &fromSock,
                        const QModelIndex &toSock,
                        bool enableTransaction)
{
    return m_impl->addLink(subgIdx, fromSock, toSock, enableTransaction);
}

QModelIndex GraphsTreeModel::addLink(
                        const QModelIndex &subgIdx,
                        const EdgeInfo& info,
                        bool enableTransaction)
{
    return m_impl->addLink(info, enableTransaction);
}

void GraphsTreeModel::removeLink(const QModelIndex &linkIdx, bool enableTransaction)
{
    m_impl->removeLink(linkIdx, enableTransaction);
}

void GraphsTreeModel::removeLink(const QModelIndex &subgIdx, const EdgeInfo &linkIdx, bool enableTransaction)
{
    m_impl->removeLink(subgIdx, linkIdx, enableTransaction);
}

void GraphsTreeModel::removeSubGraph(const QString &name)
{
    ZASSERT_EXIT(m_pSubgraphs);
    m_pSubgraphs->removeSubGraph(name);
}

QModelIndex GraphsTreeModel::extractSubGraph(
                            const QModelIndexList &nodes,
                            const QModelIndexList &links,
                            const QModelIndex &fromSubg,
                            const QString &toSubg,
                            bool enableTrans)
{
    ZASSERT_EXIT(m_impl, QModelIndex());
    return m_impl->extractSubGraph(nodes, links, fromSubg, toSubg, enableTrans);
}

bool GraphsTreeModel::IsSubGraphNode(const QModelIndex &nodeIdx) const
{
    return m_impl->IsSubGraphNode(nodeIdx);
}

QModelIndex GraphsTreeModel::fork(const QModelIndex &subgIdx, const QModelIndex &subnetNodeIdx)
{
    //no need to fork anymore.
    return QModelIndex();
}

QList<EdgeInfo> GraphsTreeModel::addSubnetNode(
            IGraphsModel *pSubgraphs,
            const QString &subnetName,
            const QString &ident,
            const QString &customName)
{
    return m_impl->addSubnetNode(pSubgraphs, subnetName, ident, customName);
}

void GraphsTreeModel::updateParamInfo(
                        const QString &id,
                        PARAM_UPDATE_INFO info,
                        const QModelIndex &subGpIdx,
                        bool enableTransaction)
{
    return m_impl->updateParamInfo(id, info, subGpIdx, enableTransaction);
}

void GraphsTreeModel::updateSocketDefl(
                        const QString& id,
                        PARAM_UPDATE_INFO info,
                        const QModelIndex& subGpIdx,
                        bool enableTransaction)
{
    return m_impl->updateSocketDefl(id, info, subGpIdx, enableTransaction);
}

void GraphsTreeModel::updateNodeStatus(
                        const QString &nodeid,
                        STATUS_UPDATE_INFO info,
                        const QModelIndex &subgIdx,
                        bool enableTransaction)
{
    return m_impl->updateNodeStatus(nodeid, info, subgIdx, enableTransaction);
}

void GraphsTreeModel::updateBlackboard(
                        const QString &id,
                        const QVariant &blackboard,
                        const QModelIndex &subgIdx,
                        bool enableTransaction)
{
    return m_impl->updateBlackboard(id, blackboard, subgIdx, enableTransaction);
}

NODE_DATA GraphsTreeModel::itemData(const QModelIndex &index, const QModelIndex &subGpIdx) const
{
    return m_impl->itemData(index, subGpIdx);
}

void GraphsTreeModel::exportSubgraph(const QModelIndex& subGpIdx, NODES_DATA& nodes, LINKS_DATA& links) const
{
    return m_impl->exportSubgraph(subGpIdx, nodes, links);
}

void GraphsTreeModel::setName(const QString &name, const QModelIndex &subGpIdx)
{
    return m_impl->setName(name, subGpIdx);
}

void GraphsTreeModel::clearSubGraph(const QModelIndex &subGpIdx)
{
    ZASSERT_EXIT(m_pSubgraphs);
    m_pSubgraphs->clearSubGraph(subGpIdx);
}

void GraphsTreeModel::clear()
{
    m_impl->clear();
    emit modelClear();
}

void GraphsTreeModel::undo()
{
    ApiLevelScope batch(this);
    m_stack->undo();
}

void GraphsTreeModel::redo()
{
    ApiLevelScope batch(this);
    m_stack->redo();
}

void GraphsTreeModel::newSubgraph(const QString &graphName)
{
    ZASSERT_EXIT(m_pSubgraphs);
    m_pSubgraphs->newSubgraph(graphName);
}

void GraphsTreeModel::initMainGraph()
{
    m_impl->initMainGraph();
}

void GraphsTreeModel::renameSubGraph(const QString &oldName, const QString &newName)
{
    ZASSERT_EXIT(m_impl);
    m_impl->renameSubGraph(oldName, newName);
}

bool GraphsTreeModel::isDirty() const
{
    return m_dirty;
}

QModelIndexList GraphsTreeModel::subgraphsIndice() const
{
    return QModelIndexList();
}

QList<SEARCH_RESULT> GraphsTreeModel::search(
                            const QString &content,
                            int searchType,
                            int searchOpts)
{
    return m_impl->search(content, searchType, searchOpts);
}

QModelIndexList GraphsTreeModel::searchInSubgraph(
                            const QString& objName,
                            const QModelIndex& idx)
{
    return m_impl->searchInSubgraph(objName, idx);
}

void GraphsTreeModel::removeGraph(int idx)
{
    //todo:
}

QRectF GraphsTreeModel::viewRect(const QModelIndex &subgIdx)
{
    return m_impl->viewRect(subgIdx);
}

void GraphsTreeModel::markDirty()
{
    m_dirty = true;
    emit dirtyChanged();
}

void GraphsTreeModel::clearDirty()
{
    m_dirty = false;
    emit dirtyChanged();
}

void GraphsTreeModel::collaspe(const QModelIndex &subgIdx)
{
    return m_impl->collaspe(subgIdx);
}

void GraphsTreeModel::expand(const QModelIndex &subgIdx)
{
    return m_impl->expand(subgIdx);
}

void GraphsTreeModel::setIOProcessing(bool bIOProcessing)
{
    m_bIOProcessing = bIOProcessing;
}

bool GraphsTreeModel::IsIOProcessing() const
{
    return m_bIOProcessing;
}

void GraphsTreeModel::beginTransaction(const QString& name)
{
    m_stack->beginMacro(name);
    beginApiLevel();
}

void GraphsTreeModel::endTransaction()
{
    m_stack->endMacro();
    endApiLevel();
}

void GraphsTreeModel::beginApiLevel()
{
    if (IsIOProcessing() || !isApiRunningEnable())
        return;

    //todo: Thread safety
    m_apiLevel++;
}

void GraphsTreeModel::endApiLevel()
{
    if (IsIOProcessing() || !isApiRunningEnable())
        return;

    m_apiLevel--;
    if (m_apiLevel == 0) {
        emit apiBatchFinished();
    }
}

LinkModel* GraphsTreeModel::linkModel(const QModelIndex &subgIdx) const
{
    return m_impl->linkModel(subgIdx);
}

QModelIndexList GraphsTreeModel::findSubgraphNode(const QString& subgName)
{
    return QModelIndexList();
}

int GraphsTreeModel::ModelSetData(
                        const QPersistentModelIndex &idx,
                        const QVariant &value,
                        int role,
                        const QString &comment)
{
    return m_impl->ModelSetData(idx, value, role, comment);
}

QAbstractItemModel *GraphsTreeModel::implModel() {
    return m_impl;
}

int GraphsTreeModel::undoRedo_updateSubgDesc(const QString &descName, const NODE_DESC &desc)
{
    ZASSERT_EXIT(m_pSubgraphs, 0);
    return m_pSubgraphs->undoRedo_updateSubgDesc(descName, desc);
}

QModelIndex GraphsTreeModel::indexFromPath(const QString &path)
{
    //subgraph need path?
    return m_impl->indexFromPath(path);
}

bool GraphsTreeModel::addExecuteCommand(QUndoCommand *pCommand)
{
    //toask: need level?
    if (!pCommand)
        return false;
    m_stack->push(pCommand);
    return 1;
}

void GraphsTreeModel::setIOVersion(zenoio::ZSG_VERSION ver)
{
    m_version = ver;
}

zenoio::ZSG_VERSION GraphsTreeModel::ioVersion() const
{
    return m_version;
}

void GraphsTreeModel::setApiRunningEnable(bool bEnable)
{
    m_bApiEnableRun = bEnable;
}

bool GraphsTreeModel::isApiRunningEnable() const
{
    return m_bApiEnableRun;
}

bool GraphsTreeModel::setCustomName(const QModelIndex &subgIdx, const QModelIndex &idx, const QString &value)
{
    return m_impl->setCustomName(subgIdx, idx, value);
}

void GraphsTreeModel::onSubgrahSync(const QModelIndex& subgIdx) {
    ZASSERT_EXIT(m_impl);
    m_impl->onSubgrahSync(subgIdx);
}

void GraphsTreeModel::markNodeDataChanged(const QModelIndex&)
{
    //NO_IMPL
}

void GraphsTreeModel::clearNodeDataChanged()
{
    //NO_IMPL
}
