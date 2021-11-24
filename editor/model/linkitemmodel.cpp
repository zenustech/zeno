#include "linkitemmodel.h"

LinkItemModel::LinkItemModel(QObject* parent = nullptr)
{

}

LinkItemModel::~LinkItemModel()
{

}

QModelIndex LinkItemModel::index(int row, int column, const QModelIndex& parent) const
{
	//TODO
	return QModelIndex();
}

QModelIndex LinkItemModel::index(QString id, const QModelIndex& parent)
{
	auto it = datas.find(id);
	if (it == datas.end())
		return QModelIndex();
	return createIndex(it - datas.begin(), 0, &it->second);
}

QModelIndex LinkItemModel::parent(const QModelIndex& child) const
{
	//TODO
	return QModelIndex();
}

int LinkItemModel::rowCount(const QModelIndex& parent) const
{
	return datas.size();
}

int LinkItemModel::columnCount(const QModelIndex& parent)
{
	return 1;
}

QVariant LinkItemModel::data(const QModelIndex& index, int role)
{
	LinkItem* pItem = reinterpret_cast<LinkItem*>(index.internalPointer());
	switch (role)
	{
	case ROLE_OBJID: return pItem->id;
	case ROLE_SRCNODE: return pItem->srcNodeId;
	case ROLE_DSTNODE: return pItem->dstNodeId;
	case ROLE_PARAMETERS: return QVariant();
	default:
		return QVariant();
	}
}

bool LinkItemModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	//when to use it?
	return false;
}