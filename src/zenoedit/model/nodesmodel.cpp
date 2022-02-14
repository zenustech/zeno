#include "nodesmodel.h"
#include <zenoui/model/modelrole.h>
#include "nodeitem.h"
#include <zeno/utils/log.h>


NodesModel::NodesModel(QObject *parent)
    : m_rootItem(new NodeItem)
{
    m_rootItem->setData("ROOT", ROLE_OBJID);
}

NodesModel::~NodesModel()
{
}

QModelIndex NodesModel::index(int row, int column, const QModelIndex& parent) const
{
    SP_NODE_ITEM parentItem = itemFromIndex(parent);
    if (parentItem == nullptr || row >= parentItem->childrenCount() || column != 0)
        return QModelIndex();
    SP_NODE_ITEM childItem = parentItem->child(row);
    return createIndex(row, column, &childItem);
}

QModelIndex NodesModel::index(QString id, const QModelIndex& parent) const
{
    SP_NODE_ITEM parentItem = itemFromIndex(parent);
    if (parentItem == nullptr) return QModelIndex();

    SP_NODE_ITEM pItem = parentItem->child(id);
    int row = parentItem->indexOfItem(pItem);
    return createIndex(row, 0, &pItem);
}

void NodesModel::appendItem(SP_NODE_ITEM pItem)
{
    m_rootItem->appendItem(pItem);
}

SP_NODE_ITEM NodesModel::rootItem() const
{
    return m_rootItem;
}

QModelIndex NodesModel::parent(const QModelIndex& child) const
{
    SP_NODE_ITEM pItem = itemFromIndex(child);
    if (!pItem)
        return QModelIndex();

    SP_NODE_ITEM parentItem = pItem->parent();
    if (!parentItem)
        return QModelIndex();

    SP_NODE_ITEM pparentItem = parentItem->parent();
    if (!pparentItem)
        return QModelIndex();

    int index = pparentItem->indexOfItem(parentItem);
    return createIndex(index, 0, &parentItem);
}

int NodesModel::rowCount(const QModelIndex& parent) const
{
    SP_NODE_ITEM parentItem = itemFromIndex(parent);
    if (!parentItem) return 0;
    return parentItem->childrenCount();
}

int NodesModel::columnCount(const QModelIndex& parent) const
{
	return 1;
}

QVariant NodesModel::data(const QModelIndex& index, int role) const
{
    SP_NODE_ITEM pItem = itemFromIndex(index);
    return pItem->data(role);
}

bool NodesModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    SP_NODE_ITEM pItem = itemFromIndex(index);
    if (!pItem)
        return false;

    pItem->setData(value, role);
    return true;
}

bool NodesModel::hasChildren(const QModelIndex& parent) const
{
    SP_NODE_ITEM pItem = itemFromIndex(parent);
    return pItem && pItem->childrenCount() > 0;
}

QVariant NodesModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    return _base::headerData(section, orientation, role);
}

bool NodesModel::setHeaderData(int section, Qt::Orientation orientation, const QVariant &value, int role)
{
    return _base::setHeaderData(section, orientation, value, role);
}

QMap<int, QVariant> NodesModel::itemData(const QModelIndex& index) const
{
    SP_NODE_ITEM pItem = itemFromIndex(index);
    return QMap(pItem->m_datas);
}

bool NodesModel::setItemData(const QModelIndex& index, const QMap<int, QVariant>& roles)
{
    SP_NODE_ITEM pItem = itemFromIndex(index);
    if (!pItem)
        return false;
    pItem->m_datas = roles.toStdMap();
    return true;
}

QModelIndexList NodesModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QHash<int, QByteArray> NodesModel::roleNames() const
{
    return _base::roleNames();
}

SP_NODE_ITEM NodesModel::itemFromIndex(const QModelIndex& index) const
{
    SP_NODE_ITEM* pItem = reinterpret_cast<SP_NODE_ITEM *>(index.internalPointer());
    return *pItem;
}

bool NodesModel::insertRow(int row, SP_NODE_ITEM pItem, const QModelIndex &parent)
{
    //TODO: begin/endInsertRows
    SP_NODE_ITEM parentItem = itemFromIndex(parent);
    if (!parentItem)
        return false;
    
    parentItem->insertItem(row, pItem);
    return true;
}

bool NodesModel::removeRow(int row, const QModelIndex &parent)
{
    SP_NODE_ITEM parentItem = itemFromIndex(parent);
    if (!parentItem)
        return false;

    parentItem->removeItem(row);
    return true;
}
