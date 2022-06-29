#include "command.h"
#include "graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "util/log.h"


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
        it->second.linkIndice.clear();
    for (auto it = inputs.begin(); it != inputs.end(); it++)
        it->second.linkIndice.clear();
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


AddLinkCommand::AddLinkCommand(EdgeInfo info, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_info(info)
	, m_model(pModel)
	, m_subgIdx(subgIdx)
{
}

void AddLinkCommand::redo()
{
    QModelIndex idx = m_model->addLink(m_info, m_subgIdx);
    ZASSERT_EXIT(idx.isValid());
	m_linkIdx = QPersistentModelIndex(idx);
}

void AddLinkCommand::undo()
{
    m_model->removeLink(m_linkIdx, m_subgIdx);
}


RemoveLinkCommand::RemoveLinkCommand(QPersistentModelIndex linkIdx, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_linkIdx(linkIdx)
    , m_model(pModel)
    , m_subgIdx(subgIdx)
{
    m_info.inputNode = linkIdx.data(ROLE_INNODE).toString();
    m_info.inputSock = linkIdx.data(ROLE_INSOCK).toString();
    m_info.outputNode = linkIdx.data(ROLE_OUTNODE).toString();
    m_info.outputSock = linkIdx.data(ROLE_OUTSOCK).toString();
}

void RemoveLinkCommand::redo()
{
    m_model->removeLink(m_linkIdx, m_subgIdx);
}

void RemoveLinkCommand::undo()
{
	QModelIndex idx = m_model->addLink(m_info, m_subgIdx);
    ZASSERT_EXIT(idx.isValid());
	m_linkIdx = QPersistentModelIndex(idx);
}


UpdateDataCommand::UpdateDataCommand(const QString& nodeid, const PARAM_UPDATE_INFO& updateInfo, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_nodeid(nodeid)
    , m_updateInfo(updateInfo)
    , m_subgIdx(subgIdx)
    , m_model(pModel)
{
}

void UpdateDataCommand::redo()
{
    m_model->updateParamInfo(m_nodeid, m_updateInfo, m_subgIdx);
}

void UpdateDataCommand::undo()
{
    PARAM_UPDATE_INFO revertInfo;
    revertInfo.name = m_updateInfo.name;
    revertInfo.newValue = m_updateInfo.oldValue;
    revertInfo.oldValue = m_updateInfo.newValue;
    m_model->updateParamInfo(m_nodeid, revertInfo, m_subgIdx);
}


UpdateSockDeflCommand::UpdateSockDeflCommand(const QString& nodeid, const PARAM_UPDATE_INFO& updateInfo, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_nodeid(nodeid)
    , m_updateInfo(updateInfo)
    , m_subgIdx(subgIdx)
    , m_model(pModel)
{
}

void UpdateSockDeflCommand::redo()
{
    m_model->updateSocketDefl(m_nodeid, m_updateInfo, m_subgIdx);
}

void UpdateSockDeflCommand::undo()
{
    PARAM_UPDATE_INFO revertInfo;
    revertInfo.name = m_updateInfo.name;
    revertInfo.newValue = m_updateInfo.oldValue;
    revertInfo.oldValue = m_updateInfo.newValue;
    m_model->updateSocketDefl(m_nodeid, revertInfo, m_subgIdx);
}


UpdateStateCommand::UpdateStateCommand(const QString& nodeid, STATUS_UPDATE_INFO info, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : m_nodeid(nodeid)
    , m_info(info)
    , m_pModel(pModel)
    , m_subgIdx(subgIdx)
{
}

void UpdateStateCommand::redo()
{
    m_pModel->updateNodeStatus(m_nodeid, m_info, m_subgIdx);
}

void UpdateStateCommand::undo()
{
    STATUS_UPDATE_INFO info;
    info.role = m_info.role;
    info.newValue = m_info.oldValue;
    info.oldValue = m_info.newValue;
    m_pModel->updateNodeStatus(m_nodeid, info, m_subgIdx);
}


UpdateSocketCommand::UpdateSocketCommand(const QString& nodeid, SOCKET_UPDATE_INFO info, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : m_nodeid(nodeid)
    , m_info(info)
    , m_pModel(pModel)
    , m_subgIdx(subgIdx)
{
}

void UpdateSocketCommand::redo()
{
    m_pModel->updateSocket(m_nodeid, m_info, m_subgIdx);
}

void UpdateSocketCommand::undo()
{
    SOCKET_UPDATE_INFO revertInfo;
    revertInfo.bInput = m_info.bInput;
    revertInfo.newInfo = m_info.oldInfo;
    revertInfo.oldInfo = m_info.newInfo;
    switch (m_info.updateWay)
    {
    case SOCKET_INSERT:
        revertInfo.updateWay = SOCKET_REMOVE;
        break;
    case SOCKET_REMOVE:
        revertInfo.updateWay = SOCKET_INSERT;
        break;
    default:
        revertInfo.updateWay = m_info.updateWay;
        break;
    }
    m_pModel->updateSocket(m_nodeid, revertInfo, m_subgIdx);
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


UpdateNotDescSockNameCommand::UpdateNotDescSockNameCommand(const QString& nodeid, const SOCKET_UPDATE_INFO& updateInfo, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : m_nodeid(nodeid)
    , m_info(updateInfo)
    , m_pModel(pModel)
    , m_subgIdx(subgIdx)
{
}

void UpdateNotDescSockNameCommand::redo()
{
    m_pModel->updateSocketNameNotDesc(m_nodeid, m_info, m_subgIdx, false);
}

void UpdateNotDescSockNameCommand::undo()
{
    SOCKET_UPDATE_INFO revertInfo;
    revertInfo.bInput = m_info.bInput;
    revertInfo.newInfo = m_info.oldInfo;
    revertInfo.oldInfo = m_info.newInfo;
    switch (m_info.updateWay)
    {
    case SOCKET_INSERT:
        revertInfo.updateWay = SOCKET_REMOVE;
        break;
    case SOCKET_REMOVE:
        revertInfo.updateWay = SOCKET_INSERT;
        break;
    default:
        revertInfo.updateWay = m_info.updateWay;
        break;
    }
    m_pModel->updateSocketNameNotDesc(m_nodeid, revertInfo, m_subgIdx, false);
}


ImportNodesCommand::ImportNodesCommand(const QMap<QString, NODE_DATA>& nodes, QPointF pos, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : m_nodes(nodes)
    , m_model(pModel)
    , m_subgIdx(subgIdx)
    , m_pos(pos)
{
}

void ImportNodesCommand::redo()
{
    m_model->importNodes(m_nodes, m_pos, m_subgIdx, false);
}

void ImportNodesCommand::undo()
{
    for (QString id : m_nodes.keys())
    {
        m_model->removeNode(id, m_subgIdx, false);
    }
}