#include "command.h"
#include "subgraphmodel.h"
#include "graphsmodel.h"


AddNodeCommand::AddNodeCommand(int row, const QString& id, const NODE_DATA& data, GraphsModel* pModel, QPersistentModelIndex subgIdx)
    : QUndoCommand()
    , m_row(row)
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


RemoveNodeCommand::RemoveNodeCommand(int row, const NODE_DATA& data, SubGraphModel* pModel)
    : QUndoCommand()
    , m_row(row)
    , m_data(data)
    , m_model(pModel)
{
}

RemoveNodeCommand::~RemoveNodeCommand()
{
}

void RemoveNodeCommand::redo()
{
    m_model->removeNode(m_row);
}

void RemoveNodeCommand::undo()
{
    m_model->insertRow(m_row, m_data);
}


AddRemoveLinkCommand::AddRemoveLinkCommand(EdgeInfo info, bool bAdded, SubGraphModel *pModel)
    : QUndoCommand()
    , m_info(info)
    , m_bAdded(bAdded)
    , m_model(pModel)
{
}

void AddRemoveLinkCommand::redo()
{
    /*
    if (m_bAdded)
        m_model->addLink(m_info);
    else
        m_model->removeLink(m_info);
    */
}

void AddRemoveLinkCommand::undo()
{
    /*
    if (m_bAdded)
        m_model->removeLink(m_info);
    else
        m_model->addLink(m_info);
    */
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