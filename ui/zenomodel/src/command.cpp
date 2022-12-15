#include "command.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include "zassert.h"
#include "apilevelscope.h"


AddNodeCommand::AddNodeCommand(const QString& id, const NODE_DATA& data, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_id(id)
    , m_model(pModel)
    , m_data(data)
    , m_subgIdx(subgIdx)
{
}

AddNodeCommand::~AddNodeCommand()
{
}

void AddNodeCommand::redo()
{
    m_model->addNode(m_data, m_subgIdx);
}

void AddNodeCommand::undo()
{
    m_model->removeNode(m_id, m_subgIdx);
}


RemoveNodeCommand::RemoveNodeCommand(int row, NODE_DATA data, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_data(data)
    , m_model(pModel)
    , m_subgIdx(subgIdx)
    , m_row(row)
{
    m_id = m_data[ROLE_OBJID].toString();

    //all links will be removed when remove node, for caching other type data,
    //we have to clean the data here.
    OUTPUT_SOCKETS outputs = m_data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    INPUT_SOCKETS inputs = m_data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    for (auto it = outputs.begin(); it != outputs.end(); it++)
    {
        it->second.info.links.clear();
    }
    for (auto it = inputs.begin(); it != inputs.end(); it++)
    {
        it->second.info.links.clear();
    }
    m_data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    m_data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
}

RemoveNodeCommand::~RemoveNodeCommand()
{
}

void RemoveNodeCommand::redo()
{
    m_model->removeNode(m_id, m_subgIdx);
}

void RemoveNodeCommand::undo()
{
    m_model->addNode(m_data, m_subgIdx);
}


AddLinkCommand::AddLinkCommand(EdgeInfo info, GraphsModel* pModel)
    : QUndoCommand()
    , m_info(info)
    , m_model(pModel)
{
}

void AddLinkCommand::redo()
{
    QModelIndex idx = m_model->addLink(m_info, true);
    ZASSERT_EXIT(idx.isValid());
	m_linkIdx = QPersistentModelIndex(idx);
}

void AddLinkCommand::undo()
{
    m_model->removeLink(m_linkIdx);
}


RemoveLinkCommand::RemoveLinkCommand(QPersistentModelIndex linkIdx, GraphsModel* pModel)
    : QUndoCommand()
    , m_linkIdx(linkIdx)
    , m_model(pModel)
{
    m_info.inputNode = linkIdx.data(ROLE_INNODE).toString();
    m_info.inputSock = linkIdx.data(ROLE_INSOCK).toString();
    m_info.outputNode = linkIdx.data(ROLE_OUTNODE).toString();
    m_info.outputSock = linkIdx.data(ROLE_OUTSOCK).toString();
}

void RemoveLinkCommand::redo()
{
    m_model->removeLink(m_linkIdx);
}

void RemoveLinkCommand::undo()
{
    QModelIndex idx = m_model->addLink(m_info, true);
    ZASSERT_EXIT(idx.isValid());
    m_linkIdx = QPersistentModelIndex(idx);
}


UpdateBlackboardCommand::UpdateBlackboardCommand(
        const QString& nodeid,
        BLACKBOARD_INFO newInfo,
        BLACKBOARD_INFO oldInfo,
        GraphsModel* pModel,
        QPersistentModelIndex subgIdx) 
    : m_nodeid(nodeid)
    , m_oldInfo(oldInfo)
    , m_newInfo(newInfo)
    , m_pModel(pModel)
    , m_subgIdx(subgIdx)
{
}

void UpdateBlackboardCommand::redo()
{
    m_pModel->updateBlackboard(m_nodeid, m_newInfo, m_subgIdx, false);
}

void UpdateBlackboardCommand::undo()
{
    m_pModel->updateBlackboard(m_nodeid, m_oldInfo, m_subgIdx, false);
}


ImportNodesCommand::ImportNodesCommand(
                const QMap<QString, NODE_DATA>& nodes,
                const QList<EdgeInfo>& links,
                QPointF pos,
                GraphsModel* pModel,
                QPersistentModelIndex subgIdx)
    : m_nodes(nodes)
    , m_links(links)
    , m_model(pModel)
    , m_subgIdx(subgIdx)
    , m_pos(pos)
{
}

void ImportNodesCommand::redo()
{
    m_model->importNodes(m_nodes, m_links, m_pos, m_subgIdx, false);
}

void ImportNodesCommand::undo()
{
    for (QString id : m_nodes.keys())
    {
        m_model->removeNode(id, m_subgIdx, false);
    }
}


ModelDataCommand::ModelDataCommand(IGraphsModel* pModel, const QModelIndex& idx, const QVariant& oldData, const QVariant& newData, int role)
    : m_model(pModel)
    , m_oldData(oldData)
    , m_newData(newData)
    , m_role(role)
    , m_index(idx)
{
    m_objPath = idx.data(ROLE_OBJPATH).toString();
}

void ModelDataCommand::ensureIdxValid()
{
    if (!m_index.isValid())
    {
        m_index = m_model->indexFromPath(m_objPath);    //restore the index because the index may be invalid after new/delete item.
    }
}

void ModelDataCommand::redo()
{
    ensureIdxValid();
    //todo: setpos case.
    ApiLevelScope scope(m_model);
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    if (pModel)
        pModel->setData(m_index, m_newData, m_role);
}

void ModelDataCommand::undo()
{
    ensureIdxValid();
    ApiLevelScope scope(m_model);
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    if (pModel)
        pModel->setData(m_index, m_oldData, m_role);
}


UpdateSubgDescCommand::UpdateSubgDescCommand(IGraphsModel *pModel, const QString &subgraphName, const NODE_DESC newDesc)
    : m_model(pModel)
    , m_subgName(subgraphName)
    , m_newDesc(newDesc)
{
    m_model->getDescriptor(m_subgName, m_oldDesc);
}

void UpdateSubgDescCommand::redo()
{
    m_model->updateSubgDesc(m_subgName, m_newDesc);
}

void UpdateSubgDescCommand::undo()
{
    m_model->updateSubgDesc(m_subgName, m_oldDesc);
}
