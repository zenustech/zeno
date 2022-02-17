#include "command.h"
#include "subgraphmodel.h"
#include "graphsmodel.h"
#include "modelrole.h"


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
    m_model->insertRow(m_row, m_data, m_subgIdx);
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
	Q_ASSERT(idx.isValid());
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
	Q_ASSERT(idx.isValid());
	m_linkIdx = QPersistentModelIndex(idx);
}


UpdateDataCommand::UpdateDataCommand(const QString& nodeid, const QString& paramName, const QVariant& newValue, SubGraphModel* pModel)
    : QUndoCommand()
    , m_nodeid(nodeid)
    , m_name(paramName)
    , m_newValue(newValue)
    , m_model(pModel)
{
    //should get from data(param).
    m_oldValue = m_model->getParamValue(nodeid, paramName);
}

void UpdateDataCommand::redo()
{
    m_model->updateParam(m_nodeid, m_name, m_newValue);
}

void UpdateDataCommand::undo()
{
    m_model->updateParam(m_nodeid, m_name, m_oldValue);
}


UpdateStateCommand::UpdateStateCommand(const QString& nodeid, int role, const QVariant& val, SubGraphModel* pModel)
    : m_nodeid(nodeid)
    , m_role(role)
    , m_value(val)
    , m_pModel(pModel)
{
    QModelIndex idx = m_pModel->index(nodeid);
    m_oldValue = m_pModel->data(idx, role);
}

void UpdateStateCommand::redo()
{
    m_pModel->updateNodeState(m_nodeid, m_role, m_value);
}

void UpdateStateCommand::undo()
{
    m_pModel->updateNodeState(m_nodeid, m_role, m_oldValue);
}