#include "graphsmodel.h"
#include "subgraphmodel.h"
#include "modelrole.h"
#include "modeldata.h"


SubGraphModel::SubGraphModel(GraphsModel* pGraphsModel, QObject *parent)
    : QAbstractItemModel(parent)
    , m_pGraphsModel(pGraphsModel)
    , m_stack(new QUndoStack(this))
{
    //to think: should graphsModel be a parent of this.
}

SubGraphModel::~SubGraphModel()
{
}

SubGraphModel::SubGraphModel(const SubGraphModel &rhs)
    : m_pGraphsModel(nullptr)
    , m_stack(new QUndoStack(this))
    , m_key2Row(rhs.m_key2Row)
    , m_row2Key(rhs.m_row2Key)
    , m_rect(rhs.m_rect)
    , m_name(rhs.m_name)
    , m_nodes(rhs.m_nodes)
{
}

NODE_DESCS SubGraphModel::descriptors()
{
    return m_pGraphsModel->descriptors();
}

NODES_DATA SubGraphModel::nodes()
{
    NODES_DATA datas;
    for (auto iter = m_nodes.keyValueBegin(); iter != m_nodes.keyValueEnd(); iter++)
    {
        datas[iter->first] = iter->second;
    }
    return datas;
}

void SubGraphModel::clear()
{
    m_nodes.clear();
    m_key2Row.clear();
    m_row2Key.clear();
    emit clearLayout();
}

void SubGraphModel::reload()
{
    emit reloaded();
}

QUndoStack* SubGraphModel::undoStack() const
{
    return m_stack;
}

QModelIndex SubGraphModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount())
        return QModelIndex();

    auto itRow = m_row2Key.find(row);
    Q_ASSERT(itRow != m_row2Key.end());

    auto itItem = m_nodes.find(itRow.value());
    Q_ASSERT(itItem != m_nodes.end());

    return createIndex(row, 0, nullptr);
}

QModelIndex SubGraphModel::index(QString id, const QModelIndex& parent) const
{
    auto it = m_nodes.find(id);
    if (it == m_nodes.end())
        return QModelIndex();

    int row = m_key2Row[id];
    return createIndex(row, 0, nullptr);
}

void SubGraphModel::appendItem(const NODE_DATA& nodeData, bool enableTransaction)
{
    int nRow = m_nodes.size();
    if (enableTransaction)
    {
        QString id = nodeData[ROLE_OBJID].toString();
        AddNodeCommand *pCmd = new AddNodeCommand(nRow, id, nodeData, this);
        m_stack->push(pCmd);
    }
    else
    {
        insertRow(nRow, nodeData);
    }
}

void SubGraphModel::appendNodes(const QList<NODE_DATA>& nodes, bool enableTransaction)
{
    m_stack->beginMacro("add nodes");
    //add nodes.
    for (auto node : nodes)
    {
        appendItem(node, true);
    }

    for (const NODE_DATA& node : nodes)
    {
        INPUT_SOCKETS inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
        QString inNode = node[ROLE_OBJID].toString();
        for (INPUT_SOCKET inSock : inputs)
        {
            for (QString outNode : inSock.outNodes.keys())
            {
                for (SOCKET_INFO outSock : inSock.outNodes[outNode])
                {
                    addLink(EdgeInfo(outNode, inNode, outSock.name, inSock.info.name), true);
                }
            }
        }
    }
    m_stack->endMacro();
}

void SubGraphModel::removeNode(const QString& nodeid, bool enableTransaction)
{
    Q_ASSERT(m_key2Row.find(nodeid) != m_key2Row.end());
    int row = m_key2Row[nodeid];
    if (enableTransaction)
    {
        m_stack->beginMacro("remove single node");
        RemoveNodeCommand *pCmd = new RemoveNodeCommand(row, m_nodes[nodeid], this);
        m_stack->push(pCmd);
        m_stack->endMacro();
    }
    else
    {
        removeRows(row, 0);
    }
}

void SubGraphModel::removeNode(int row, bool enableTransaction)
{
    removeNode(m_row2Key[row], enableTransaction);
}

bool SubGraphModel::_removeRow(const QModelIndex& index)
{
    //remove node by id and update params from other node.
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return false;

    QString currNode = index.data(ROLE_OBJID).toString();

    const INPUT_SOCKETS& inputs = nodeData[ROLE_INPUTS].value<INPUT_SOCKETS>();
    for (QString inSock : inputs.keys())
    {
        for (QString outNode : inputs[inSock].outNodes.keys())
        {
            SOCKETS_INFO outSocks = inputs[inSock].outNodes[outNode];
            for (QString outSock : outSocks.keys())
            {
                EdgeInfo info(outNode, currNode, outSock, inSock);
                removeLink(info, true);
            }
        }
    }

    // in this loop, output refers to current node's output, input refers to what output points to.
    const OUTPUT_SOCKETS& outputs = index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    for (QString outSock : outputs.keys())
    {
        for (QString inNode : outputs[outSock].inNodes.keys())
        {
            SOCKETS_INFO sockets = outputs[outSock].inNodes[inNode];
            for (QString inSock : sockets.keys())
            {
                EdgeInfo info(currNode, inNode, outSock, inSock);
                removeLink(info, true);
            }
        }
    }

    int row = index.row();
    QString id = m_row2Key[row];
    Q_ASSERT(!id.isEmpty());
    for (int r = row + 1; r < rowCount(); r++) {
        const QString &key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }

    m_row2Key.remove(rowCount() - 1);
    m_key2Row.remove(id);
    m_nodes.remove(id);
    return true;
}

void SubGraphModel::addLink(const EdgeInfo& info, bool enableTransaction)
{
    if (enableTransaction)
    {
        AddRemoveLinkCommand *pCmd = new AddRemoveLinkCommand(info, true, this);
        m_stack->push(pCmd);
    }
    else
    {
        _addLink(info);
    }
}

void SubGraphModel::_addLink(const EdgeInfo& info)
{
    const QModelIndex &outIdx = this->index(info.outputNode);
    OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    outputs[info.outputSock].inNodes[info.inputNode][info.inputSock] = SOCKET_INFO(info.inputNode, info.inputSock);
    setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);

    const QModelIndex &inIdx = this->index(info.inputNode);
    INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    inputs[info.inputSock].outNodes[info.outputNode][info.outputSock] = SOCKET_INFO(info.outputNode, info.outputSock);
    setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);

    emit linkChanged(true, info.outputNode, info.outputSock, info.inputNode, info.inputSock);
}

void SubGraphModel::removeLink(const EdgeInfo& info, bool enableTransaction)
{
    if (enableTransaction)
    {
        m_stack->beginMacro("remove one link");
        AddRemoveLinkCommand* pCmd = new AddRemoveLinkCommand(info, false, this);
        m_stack->push(pCmd);
        m_stack->endMacro();
    }
    else
    {
        _removeLink(info);
    }
}

void SubGraphModel::_removeLink(const EdgeInfo &info)
{
    const QModelIndex &outIdx = this->index(info.outputNode);
    OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    auto &delInItem = outputs[info.outputSock].inNodes[info.inputNode];
    delInItem.remove(info.inputSock);
    if (delInItem.isEmpty())
        outputs[info.outputSock].inNodes.remove(info.inputNode);
    setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);

    const QModelIndex& inIdx = this->index(info.inputNode);
    INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    auto &delOutItem = inputs[info.inputSock].outNodes[info.outputNode];
    delOutItem.remove(info.outputSock);
    if (delOutItem.isEmpty())
        inputs[info.inputSock].outNodes.remove(info.outputNode);
    setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);

    emit linkChanged(false, info.outputNode, info.outputSock, info.inputNode, info.inputSock);
}

QModelIndex SubGraphModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

int SubGraphModel::rowCount(const QModelIndex& parent) const
{
    return m_nodes.size();
}

int SubGraphModel::columnCount(const QModelIndex& parent) const
{
	return 1;
}

QVariant SubGraphModel::data(const QModelIndex& index, int role) const
{
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return QVariant();
    return nodeData[role];
}

bool SubGraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        m_nodes[id][role] = value;
        emit dataChanged(index, index, QVector<int>{role});
        return true;
    }
    else
    {
        return false;
    }
}

SubGraphModel* SubGraphModel::clone(GraphsModel* parent)
{
    SubGraphModel *pClone = new SubGraphModel(*this);
    pClone->m_pGraphsModel = parent;
    return pClone;
}

void SubGraphModel::updateParam(const QString& nodeid, const QString& paramName, const QVariant& var, bool enableTransaction)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    if (enableTransaction)
    {
        UpdateDataCommand *pCmd = new UpdateDataCommand(nodeid, paramName, var, this);
        m_stack->push(pCmd);
    }
    else
    {
        PARAMS_INFO info = m_nodes[nodeid][ROLE_PARAMETERS].value<PARAMS_INFO>();
        info[paramName].value = var;
        m_nodes[nodeid][ROLE_PARAMETERS] = QVariant::fromValue(info);
        const QModelIndex& idx = index(nodeid);
        emit dataChanged(idx, idx, QVector<int>{ROLE_PARAMETERS});
        emit paramUpdated(nodeid, paramName, var);
    }
}

QVariant SubGraphModel::getParamValue(const QString& nodeid, const QString& paramName)
{
    QVariant var;
    const PARAMS_INFO info = m_nodes[nodeid][ROLE_PARAMETERS].value<PARAMS_INFO>();
    return info[paramName].value;
}

void SubGraphModel::updateNodeState(const QString& nodeid, int role, const QVariant& val, bool enableTransaction)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;
    if (enableTransaction)
    {
        UpdateStateCommand *pCmd = new UpdateStateCommand(nodeid, role, val, this);
        m_stack->push(pCmd);
    }
    else
    {
        m_nodes[nodeid][role] = val;
        const QModelIndex& idx = index(nodeid);
        emit dataChanged(idx, idx, QVector<int>{role});
    }
}

void SubGraphModel::setPos(const QString& nodeid, const QPointF& pt)
{

}

bool SubGraphModel::hasChildren(const QModelIndex& parent) const
{
    return false;
}

QVariant SubGraphModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    return _base::headerData(section, orientation, role);
}

bool SubGraphModel::setHeaderData(int section, Qt::Orientation orientation, const QVariant &value, int role)
{
    return _base::setHeaderData(section, orientation, value, role);
}

NODE_DATA SubGraphModel::itemData(const QModelIndex &index) const
{
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return NODE_DATA();
    return nodeData;
}

QModelIndexList SubGraphModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QHash<int, QByteArray> SubGraphModel::roleNames() const
{
    return _base::roleNames();
}

bool SubGraphModel::itemFromIndex(const QModelIndex &index, NODE_DATA &retNode) const
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        retNode = m_nodes[id];
        return true;
    }
    else
    {
        return false;
    }
}

bool SubGraphModel::_insertRow(int row, const NODE_DATA& nodeData, const QModelIndex &parent)
{
    //pure insert logic, without transaction and notify stuffs.
    const QString& id = nodeData[ROLE_OBJID].toString();
    const QString& name = nodeData[ROLE_OBJNAME].toString();

    Q_ASSERT(!id.isEmpty() && !name.isEmpty() && m_nodes.find(id) == m_nodes.end());
    int nRows = m_nodes.size();
    if (row == nRows)
    {
        //append
        m_nodes[id] = nodeData;
        m_row2Key[nRows] = id;
        m_key2Row[id] = nRows;
        return true;
    }
    else if (row < nRows)
    {
        auto itRow = m_row2Key.find(row);
        Q_ASSERT(itRow != m_row2Key.end());
        int nRows = rowCount();
        for (int r = nRows; r > row; r--)
        {
            const QString &key = m_row2Key[r - 1];
            m_row2Key[r] = key;
            m_key2Row[key] = r;
        }
        m_nodes[id] = nodeData;
        m_row2Key[row] = id;
        m_key2Row[id] = row;
        return true;
    }
    else
    {
        return false;
    }
}

bool SubGraphModel::insertRow(int row, const NODE_DATA &nodeData, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row);
    bool ret = _insertRow(row, nodeData);
    endInsertRows();
    return ret;
}

bool SubGraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    _removeRow(index(row, 0));
    endRemoveRows();
    return true;
}

void SubGraphModel::onParamValueChanged(const QString& nodeid, const QString& paramName, const QVariant& var)
{
    blockSignals(true);
    updateParam(nodeid, paramName, var, true);
    blockSignals(false);
}

void SubGraphModel::onDoubleClicked(const QString& nodename)
{
    m_pGraphsModel->switchSubGraph(nodename);
}

void SubGraphModel::setName(const QString& name)
{
    m_name = name;
}

void SubGraphModel::setViewRect(const QRectF& rc)
{
    m_rect = rc;
}

QString SubGraphModel::name() const
{
    return m_name;
}

void SubGraphModel::undo()
{
    m_stack->undo();
}

void SubGraphModel::redo()
{
    m_stack->redo();
}

void SubGraphModel::beginTransaction(const QString& name)
{
    m_stack->beginMacro(name);
}

void SubGraphModel::endTransaction()
{
    m_stack->endMacro();
}