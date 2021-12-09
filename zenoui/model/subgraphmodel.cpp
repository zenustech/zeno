#include "subgraphmodel.h"
#include "modelrole.h"


SubGraphModel::SubGraphModel(QObject *parent)
    : QAbstractItemModel(parent)
{
}

SubGraphModel::~SubGraphModel()
{
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
    m_name2Id.insert(std::make_pair(name, id));
    int nRow = m_nodes.size() - 1;
    m_row2Key.insert(std::make_pair(nRow, id));
    m_key2Row.insert(std::make_pair(id, nRow));
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

bool SubGraphModel::insertRow(int row, NODEITEM_PTR pItem, const QModelIndex &parent)
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
    m_name2Id[name] = id;
    return true;
}

bool SubGraphModel::removeRow(int row, const QModelIndex &parent)
{
    //todo
    if (row < 0 || row >= rowCount()) return false;

    auto iterR2K = m_row2Key.find(row);
    Q_ASSERT(iterR2K != m_row2Key.end());
    QString id = iterR2K->second;
    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString &key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }
    m_row2Key.erase(rowCount() - 1);
    m_key2Row.erase(id);
    m_nodes.erase(id);
    return true;
}

bool SubGraphModel::insertRows(int row, int count, const QModelIndex& parent)
{
    return false;
}

bool SubGraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
    beginRemoveRows(parent, row, row);
    bool ret = removeRow(row, parent);
    endRemoveRows();
    return ret;
}


void SubGraphModel::setName(const QString& name)
{
    m_name = name;
}

QString SubGraphModel::name() const
{
    return m_name;
}