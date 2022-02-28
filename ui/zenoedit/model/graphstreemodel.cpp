#include "graphstreemodel.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>


GraphsTreeModel::GraphsTreeModel(QObject* parent)
	: QStandardItemModel(parent)
	, m_model(nullptr)
{

}

GraphsTreeModel::~GraphsTreeModel()
{

}

void GraphsTreeModel::init(IGraphsModel* pModel)
{
    clear();
	m_model = qobject_cast<GraphsModel*>(pModel);
	Q_ASSERT(m_model);
    SubGraphModel* pSubModel = m_model->subGraph("main");
    QStandardItem* pItem = appendSubModel(pSubModel);
    appendRow(pItem);
}

QStandardItem* GraphsTreeModel::appendSubModel(SubGraphModel* pModel)
{
	connect(pModel, &QAbstractItemModel::dataChanged, this, &GraphsTreeModel::on_dataChanged);
	connect(pModel, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsTreeModel::on_rowsAboutToBeInserted);
	connect(pModel, &QAbstractItemModel::rowsInserted, this, &GraphsTreeModel::on_rowsInserted);
	connect(pModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsTreeModel::on_rowsAboutToBeRemoved);
	connect(pModel, &QAbstractItemModel::rowsRemoved, this, &GraphsTreeModel::on_rowsRemoved);

	QStandardItem* pItem = new QStandardItem(pModel->name());
	for (int r = 0; r < pModel->rowCount(); r++)
	{
		const QModelIndex& idx = pModel->index(r, 0);
		const QString& objName = pModel->data(idx, ROLE_OBJNAME).toString();
		const QString& objId = pModel->data(idx, ROLE_OBJID).toString();
		QStandardItem* pSubItem = nullptr;
		if (SubGraphModel* pSubModel = m_model->subGraph(objName))
		{
			pSubItem = appendSubModel(pSubModel);
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
	return nullptr;
}

void GraphsTreeModel::on_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
	//todo: rename
}

void GraphsTreeModel::on_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last)
{
}

void GraphsTreeModel::on_rowsInserted(const QModelIndex& parent, int first, int last)
{
	SubGraphModel* pModel = qobject_cast<SubGraphModel*>(sender());
	QModelIndex itemIdx = pModel->index(first, 0, parent);
	Q_ASSERT(itemIdx.isValid());

	const QString& subName = pModel->name();
	QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, subName, -1, Qt::MatchRecursive);
	Q_ASSERT(lst.size() == 1);
	QModelIndex subgIdx = lst[0];
	QStandardItem* pSubgItem = itemFromIndex(subgIdx);

	const QString& objId = itemIdx.data(ROLE_OBJID).toString();
	const QString& objName = itemIdx.data(ROLE_OBJNAME).toString();
	//objName may be a subgraph.
	QStandardItem* pSubItem = nullptr;
	if (SubGraphModel* pSubModel = m_model->subGraph(objName))
	{
		pSubItem = appendSubModel(pSubModel);
	}
	else
	{
		pSubItem = new QStandardItem(objName);
	}
	pSubItem->setData(objId, ROLE_OBJID);
	pSubItem->setData(objName, ROLE_OBJNAME);

	pSubgItem->insertRow(first, pSubItem);
}

void GraphsTreeModel::on_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
	SubGraphModel* pModel = qobject_cast<SubGraphModel*>(sender());
	QModelIndex itemIdx = pModel->index(first, 0, parent);
	Q_ASSERT(itemIdx.isValid());

	const QString& subName = pModel->name();
	QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, subName, -1, Qt::MatchRecursive);
	Q_ASSERT(lst.size() == 1);
	QModelIndex subgIdx = lst[0];
	removeRow(first, subgIdx);
}

void GraphsTreeModel::on_rowsRemoved(const QModelIndex& parent, int first, int last)
{
}