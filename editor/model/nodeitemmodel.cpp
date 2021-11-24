#include "nodeitemmodel.h"
#include "modelrole.h"


NodeItemModel::NodeItemModel(QObject* parent = nullptr)
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

QModelIndex NodeItemModel::index(QString id, const QModelIndex& parent)
{
	auto it = datas.find(id);
	if (it == datas.end())
		return QModelIndex();
	return createIndex(it - datas.begin(), 0, &it->second);
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

int NodeItemModel::columnCount(const QModelIndex& parent)
{
	return 1;
}

QVariant NodeItemModel::data(const QModelIndex& index, int role)
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