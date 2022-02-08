#include "graphstreemodel.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>


GraphsTreeModel::GraphsTreeModel(GraphsModel* pTreeModel, QObject* parent)
	: QStandardItemModel(parent)
{

}

GraphsTreeModel::~GraphsTreeModel()
{

}

void GraphsTreeModel::init(GraphsModel* pModel)
{
	clear();
	SubGraphModel* pSubModel = pModel->subGraph("main");
	QStandardItem* pItem = appendSubModel(pModel, pSubModel);
	appendRow(pItem);
}

QStandardItem* GraphsTreeModel::appendSubModel(GraphsModel* pTreeModel, SubGraphModel* pModel)
{
	QStandardItem* pItem = new QStandardItem(pModel->name());
	for (int r = 0; r < pModel->rowCount(); r++)
	{
		const QModelIndex& idx = pModel->index(r, 0);
		const QString& objName = pModel->data(idx, ROLE_OBJNAME).toString();
		const QString& objId = pModel->data(idx, ROLE_OBJID).toString();
		QStandardItem* pSubItem = nullptr;
		if (SubGraphModel* pSubModel = pTreeModel->subGraph(objName))
		{
			pSubItem = appendSubModel(pTreeModel, pSubModel);
		}
		else
		{
			pSubItem = new QStandardItem(objName);
		}
		pSubItem->setData(objName, ROLE_OBJNAME);
		pSubItem->setData(objId, ROLE_OBJID);
		pItem->appendRow(pSubItem);
	}
	pItem->setData(pModel->name(), ROLE_OBJNAME);
	return pItem;
}