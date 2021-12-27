#include "command.h"
#include "subgraphmodel.h"


AddNodeCommand::AddNodeCommand(int row, const QString& id, const NODE_DATA& data, SubGraphModel* pModel)
    : QUndoCommand()
    , m_row(row)
    , m_id(id)
    , m_model(pModel)
    , m_data(data)
{
}

void AddNodeCommand::redo()
{
    m_model->insertRow(m_row, m_data);
}

void AddNodeCommand::undo()
{
    m_model->removeNode(m_id);
}


RemoveNodeCommand::RemoveNodeCommand(int row, const NODE_DATA& data, SubGraphModel* pModel)
    : QUndoCommand()
    , m_row(row)
    , m_data(data)
    , m_model(pModel)
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
    if (m_bAdded)
        m_model->addLink(m_info);
    else
        m_model->removeLink(m_info);
}

void AddRemoveLinkCommand::undo()
{
    if (m_bAdded)
        m_model->removeLink(m_info);
    else
        m_model->addLink(m_info);
}