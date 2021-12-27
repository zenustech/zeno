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

NODE_DESCS SubGraphModel::descriptors()
{
    return m_pGraphsModel->descriptors();
}

NODES_DATA SubGraphModel::dumpGraph()
{
    NODES_DATA datas;
    for (auto node : m_nodes)
    {
        datas[node.first] = node.second->m_datas;
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
    auto itItem = m_nodes.find(itRow->second);
    Q_ASSERT(itItem != m_nodes.end());
    return createIndex(row, 0, itItem->second.get());
}

QModelIndex SubGraphModel::index(QString id, const QModelIndex& parent) const
{
    auto it = m_nodes.find(id);
    if (it == m_nodes.end())
        return QModelIndex();
    return indexFromItem(it->second.get());
}

void SubGraphModel::appendItem(NODEITEM_PTR pItem)
{
    if (!pItem)
        return;

    const QString &id = pItem->data(ROLE_OBJID).toString();
    const QString &name = pItem->data(ROLE_OBJNAME).toString();
    Q_ASSERT(!id.isEmpty() && !name.isEmpty() &&
             m_nodes.find(id) == m_nodes.end());

    m_nodes.insert(std::make_pair(id, pItem));
    int nRow = m_nodes.size() - 1;
    m_row2Key.insert(std::make_pair(nRow, id));
    m_key2Row.insert(std::make_pair(id, nRow));

    insertRows(nRow, 1);
    //m_stack->push(new AddNodeCommand(nRow, id, pItem->m_datas, this));
}

void SubGraphModel::removeNode(const QString& nodeid)
{
    removeRows(m_key2Row[nodeid], 0);
}

void SubGraphModel::removeNode(int row)
{
    removeRows(row, 1);
}

void SubGraphModel::_removeNodeItem(const QModelIndex& index)
{
    //remove node by id and update params from other node.
    PlainNodeItem* pItem = itemFromIndex(index);
    if (!pItem)
        return;

    QString currNode = index.data(ROLE_OBJID).toString();

    INPUT_SOCKETS inputs = pItem->m_datas[ROLE_INPUTS].value<INPUT_SOCKETS>();
    for (QString inSock : inputs.keys())
    {
        for (QString outNode : inputs[inSock].outNodes.keys())
        {
            SOCKETS_INFO outSocks = inputs[inSock].outNodes[outNode];
            for (QString outSock : outSocks.keys())
            {
                const QModelIndex& outIdx = this->index(outNode);
                NODEITEM_PTR outputItem = m_nodes[outNode];
                OUTPUT_SOCKETS outputs = outputItem->data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
                outputs[outSock].inNodes.remove(currNode);
                setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
                emit linkChanged(false, outNode, outSock, currNode, inSock);   //not only modify core data but emit signal to ui.
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
                const QModelIndex &inIdx = this->index(inNode);
                INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                inputs[inSock].outNodes.remove(currNode);
                setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
                emit linkChanged(false, currNode, outSock, inNode, inSock);
            }
        }
    }

    int row = index.row();
    auto iterR2K = m_row2Key.find(row);
    Q_ASSERT(iterR2K != m_row2Key.end());
    QString id = iterR2K->second;
    for (int r = row + 1; r < rowCount(); r++) {
        const QString &key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }
    m_row2Key.erase(rowCount() - 1);
    m_key2Row.erase(id);
    m_nodes.erase(id);
}

void SubGraphModel::addLink(const QString& outNode, const QString& outSock, const QString& inNode, const QString& inSock)
{
    const QModelIndex &outIdx = this->index(outNode);
    OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    outputs[outSock].inNodes[inNode][inSock] = SOCKET_INFO(inNode, inSock);
    setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);

    const QModelIndex &inIdx = this->index(inNode);
    INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    inputs[inSock].outNodes[outNode][outSock] = SOCKET_INFO(outNode, outSock);
    setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);

    emit linkChanged(true, outNode, outSock, inNode, inSock);
}

void SubGraphModel::removeLink(const QString& outputId, const QString& outputPort, const QString& inputId, const QString& inputPort)
{
    const QModelIndex& outIdx = this->index(outputId);
    OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    auto &delInItem = outputs[outputPort].inNodes[inputId];
    delInItem.remove(inputPort);
    if (delInItem.isEmpty())
        outputs[outputPort].inNodes.remove(inputId);
    setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);

    const QModelIndex& inIdx = this->index(inputId);
    INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    auto &delOutItem= inputs[inputPort].outNodes[outputId];
    delOutItem.remove(outputPort);
    if (delOutItem.isEmpty())
        inputs[inputPort].outNodes.remove(outputId);
    setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);

    emit linkChanged(false, outputId, outputPort, inputId, inputPort);
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
    PlainNodeItem *pItem = itemFromIndex(index);
    return pItem->data(role);
}

bool SubGraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    PlainNodeItem* pItem = itemFromIndex(index);
    if (!pItem)
        return false;

    pItem->setData(value, role);
    emit dataChanged(index, index, QVector<int>{role});
    return true;
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

QMap<int, QVariant> SubGraphModel::itemData(const QModelIndex& index) const
{
    PlainNodeItem* pItem = itemFromIndex(index);
    return QMap(pItem->m_datas);
}

bool SubGraphModel::setItemData(const QModelIndex& index, const QMap<int, QVariant>& roles)
{
    PlainNodeItem* pItem = itemFromIndex(index);
    if (!pItem)
        return false;
    pItem->m_datas = roles.toStdMap();
    return true;
}

QModelIndexList SubGraphModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QHash<int, QByteArray> SubGraphModel::roleNames() const
{
    return _base::roleNames();
}

PlainNodeItem* SubGraphModel::itemFromIndex(const QModelIndex &index) const
{
    PlainNodeItem *pItem = reinterpret_cast<PlainNodeItem*>(index.internalPointer());
    return pItem;
}

QModelIndex SubGraphModel::indexFromItem(PlainNodeItem* pItem) const
{
    const QString& id = pItem->data(ROLE_OBJID).toString();
    auto it = m_nodes.find(id);
    if (it == m_nodes.end())
        return QModelIndex();
    auto itRow = m_key2Row.find(id);
    Q_ASSERT(itRow != m_key2Row.end());
    return createIndex(itRow->second, 0, pItem);
}

bool SubGraphModel::_insertRow(int row, NODEITEM_PTR pItem, const QModelIndex &parent)
{
    //TODO: begin/endInsertRows
    if (!pItem)
        return false;

    const QString &id = pItem->data(ROLE_OBJID).toString();
    const QString &name = pItem->data(ROLE_OBJNAME).toString();

    Q_ASSERT(!id.isEmpty() && !name.isEmpty() && m_nodes.find(id) == m_nodes.end());

    auto itRow = m_row2Key.find(row);
    Q_ASSERT(itRow != m_row2Key.end());
    int nRows = rowCount();
    for (int r = nRows; r >= row; r--) {
        const QString &key = m_row2Key[r - 1];
        m_row2Key[r] = key;
        m_key2Row[key] = r;
    }

    m_nodes.insert(std::make_pair(id, pItem));
    m_row2Key[row] = id;
    m_key2Row[id] = row;

    QModelIndex idx = createIndex(row, 0, pItem.get());
    insertRows(row, 1, idx);
    return true;
}

bool SubGraphModel::insertRows(int row, int count, const QModelIndex& parent)
{
    beginInsertRows(parent, row, row);
    endInsertRows();
    return false;
}

bool SubGraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    _removeNodeItem(index(row, 0));
    endRemoveRows();
    return true;
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