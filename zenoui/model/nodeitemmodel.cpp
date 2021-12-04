#include "nodeitemmodel.h"
#include "modelrole.h"
#include "nodeitem.h"


NodeItemModel::NodeItemModel(QObject* parent)
{

}

NodeItemModel::~NodeItemModel()
{

}

QModelIndex NodeItemModel::index(int row, int column, const QModelIndex& parent) const
{
	//TODO
	NodeItem* parentItem = itemFromIndex(parent);
    auto it = parentItem->m_childrens.begin();
    std::advance(it, row);
	if (it == parentItem->m_childrens.end()) {
        return QModelIndex();
	}
    return createIndex(row, column, it->second);
}

QModelIndex NodeItemModel::index(QString id, const QModelIndex& parent) const
{
    auto it = m_idMapprer.find(id);
    if (it == m_idMapprer.end())
        return QModelIndex();

	NodeItem* item = it->second;
    return QModelIndex();
}

QModelIndex NodeItemModel::parent(const QModelIndex& child) const
{
	//TODO
    NodeItem* pItem = static_cast<NodeItem*>(child.internalPointer());
    NodeItem* parentItem = pItem->parent;
    if (parentItem)
    {
        int row = 0, col = 0;
        return createIndex(row, col, parentItem);
    }
	return QModelIndex();
}

int NodeItemModel::rowCount(const QModelIndex& parent) const
{
	return _base::rowCount(parent);
}

int NodeItemModel::columnCount(const QModelIndex& parent) const
{
	return 1;
}

QVariant NodeItemModel::data(const QModelIndex& index, int role) const
{
	NodeItem* pItem = reinterpret_cast<NodeItem*>(index.internalPointer());
	switch (role)
	{
	case ROLE_OBJID: return pItem->id;
	case ROLE_OBJNAME: return pItem->name;
	case ROLE_OBJRECT: return pItem->sceneRect;
	case ROLE_PARAMETERS: return QVariant();
	default:
		return QVariant();
	}
}

bool NodeItemModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    return _base::setData(index, value, role);
}

bool NodeItemModel::hasChildren(const QModelIndex& parent) const
{
    return _base::hasChildren(parent);
}

QVariant NodeItemModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    return _base::headerData(section, orientation, role);
}

bool NodeItemModel::setHeaderData(int section, Qt::Orientation orientation, const QVariant &value, int role)
{
    return _base::setHeaderData(section, orientation, value, role);
}

QMap<int, QVariant> NodeItemModel::itemData(const QModelIndex& index) const
{
    return _base::itemData(index);
}

bool NodeItemModel::setItemData(const QModelIndex& index, const QMap<int, QVariant>& roles)
{
    return _base::setItemData(index, roles);
}

QModelIndexList NodeItemModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QHash<int, QByteArray> NodeItemModel::roleNames() const
{
    return _base::roleNames();
}

NodeItem* NodeItemModel::itemFromIndex(const QModelIndex& index) const
{
    NodeItem* pItem = static_cast<NodeItem*>(index.internalPointer());
    return pItem;
}