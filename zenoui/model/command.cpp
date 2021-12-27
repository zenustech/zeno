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
    NODEITEM_PTR item(std::make_shared<PlainNodeItem>());
    item->m_datas = m_data;
    m_model->_insertRow(m_row, item);
}

void AddNodeCommand::undo()
{
    m_model->removeNode(m_id);
}