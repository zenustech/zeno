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
	return QModelIndex();
}

QModelIndex NodeItemModel::index(QString id, const QModelIndex& parent) const
{
	auto it = datas.find(id);
	if (it == datas.end())
		return QModelIndex();
	NodeItem item = it->second;
	return createIndex(0, 0, &item);
}

QModelIndex NodeItemModel::parent(const QModelIndex& child) const
{
	//TODO
	return QModelIndex();
}

int NodeItemModel::rowCount(const QModelIndex& parent) const
{
	return datas.size();
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
	//when to use it?
	return false;
}