#include "graphstreemodel.h"
#include "graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "util/log.h"


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
    ZASSERT_EXIT(m_model);
    SubGraphModel* pSubModel = m_model->subGraph("main");
    QStandardItem* pItem = appendSubModel(pSubModel);
    appendRow(pItem);

	connect(m_model, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsTreeModel::on_graphs_rowsAboutToBeRemoved);
}

QStandardItem* GraphsTreeModel::appendSubModel(SubGraphModel* pModel)
{
	connect(pModel, &QAbstractItemModel::dataChanged, this, &GraphsTreeModel::on_subg_dataChanged);
	connect(pModel, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsTreeModel::on_subg_rowsAboutToBeInserted);
	connect(pModel, &QAbstractItemModel::rowsInserted, this, &GraphsTreeModel::on_subg_rowsInserted);
	connect(pModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsTreeModel::on_subg_rowsAboutToBeRemoved);
	connect(pModel, &QAbstractItemModel::rowsRemoved, this, &GraphsTreeModel::on_subg_rowsRemoved);

	QStandardItem* pItem = new QStandardItem(pModel->name());
	pItem->setEditable(true);
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
			pSubItem->setEditable(false);
		}
		pSubItem->setData(objName, ROLE_OBJNAME);
		pSubItem->setData(objId, ROLE_OBJID);
		pItem->appendRow(pSubItem);
	}
	pItem->setData(pModel->name(), ROLE_OBJNAME);
	return pItem;
}

void GraphsTreeModel::on_subg_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
	if (roles[0] == ROLE_OBJNAME)
	{
		SubGraphModel* pModel = qobject_cast<SubGraphModel*>(sender());
		const QString& nodeId = topLeft.data(ROLE_OBJID).toString();
		const QModelIndex& modelIdx = pModel->index(nodeId);
		QModelIndexList lst = match(index(0, 0), ROLE_OBJID, nodeId, 1, Qt::MatchRecursive);
		ZASSERT_EXIT(lst.size() == 1);
		QModelIndex treeIdx = lst[0];

		const QString& oldName = treeIdx.data(ROLE_OBJNAME).toString();
		const QString& newName = modelIdx.data(ROLE_OBJNAME).toString();
		if (oldName != newName)
		{
			setData(treeIdx, newName, ROLE_OBJNAME);
			setData(treeIdx, newName, Qt::DisplayRole);
		}
	}
}

void GraphsTreeModel::on_subg_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last)
{
}

void GraphsTreeModel::on_subg_rowsInserted(const QModelIndex& parent, int first, int last)
{
	SubGraphModel* pModel = qobject_cast<SubGraphModel*>(sender());
	QModelIndex itemIdx = pModel->index(first, 0, parent);
	ZASSERT_EXIT(itemIdx.isValid());

	const QString& subName = pModel->name();
	QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, subName, -1, Qt::MatchRecursive);
	//Q_ASSERT(lst.size() == 1);
	//todo: multiple case.
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

void GraphsTreeModel::on_subg_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
	SubGraphModel* pModel = qobject_cast<SubGraphModel*>(sender());
	QModelIndex itemIdx = pModel->index(first, 0, parent);
	ZASSERT_EXIT(itemIdx.isValid());

	const QString& subName = pModel->name();
	QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, subName, -1, Qt::MatchRecursive);
	//Q_ASSERT(lst.size() == 1);
	if (lst.size() == 1)
	{
		QModelIndex subgIdx = lst[0];
		removeRow(first, subgIdx);
	}
}

void GraphsTreeModel::on_subg_rowsRemoved(const QModelIndex& parent, int first, int last)
{
}

void GraphsTreeModel::on_graphs_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
	QModelIndex subgIdx = m_model->index(first, 0, parent);
	ZASSERT_EXIT(subgIdx.isValid());

	const QString& subgName = m_model->name(subgIdx);
	QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, subgName, -1, Qt::MatchRecursive);
	for (QModelIndex idx : lst)
	{
		removeRow(idx.row(), idx.parent());
	}
}