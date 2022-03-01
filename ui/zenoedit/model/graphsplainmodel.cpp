#include "graphsplainmodel.h"
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"


GraphsPlainModel::GraphsPlainModel(QObject* parent)
	: QStandardItemModel(parent)
	, m_model(nullptr)
{

}

GraphsPlainModel::~GraphsPlainModel()
{

}

void GraphsPlainModel::init(IGraphsModel* pModel)
{
	clear();
	m_model = qobject_cast<GraphsModel*>(pModel);
	Q_ASSERT(m_model);
	for (int r = 0; r < m_model->rowCount(); r++)
	{
		const QModelIndex& idx = m_model->index(r, 0);
		QString subgName = m_model->data(idx).toString();
		QStandardItem* pItem = new QStandardItem(subgName);
		pItem->setData(subgName, ROLE_OBJNAME);
		appendRow(pItem);
	}

	connect(m_model, &QAbstractItemModel::dataChanged, this, &GraphsPlainModel::on_dataChanged);
	connect(m_model, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsPlainModel::on_rowsAboutToBeInserted);
	connect(m_model, &QAbstractItemModel::rowsInserted, this, &GraphsPlainModel::on_rowsInserted);
	connect(m_model, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsPlainModel::on_rowsAboutToBeRemoved);
	connect(m_model, &QAbstractItemModel::rowsRemoved, this, &GraphsPlainModel::on_rowsRemoved);
}

bool GraphsPlainModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (role == Qt::EditRole)
	{
		const QString& newName = value.toString();
		const QString& oldName = data(index, Qt::DisplayRole).toString();
		if (newName != oldName)
		{
			SubGraphModel* pModel = m_model->subGraph(oldName);
			if (pModel)
			{
				m_model->renameSubGraph(oldName, newName);
			}
			else
			{
				//new subgraph.
				pModel = new SubGraphModel(m_model);
				pModel->setName(newName);
				m_model->appendSubGraph(pModel);
				//to activate the view.
			}
		}
	}
	return QStandardItemModel::setData(index, value, role);
}

bool GraphsPlainModel::submit()
{
	return QStandardItemModel::submit();
}

void GraphsPlainModel::revert()
{
	//cannot know which index.
	return QStandardItemModel::revert();
}

void GraphsPlainModel::submit(const QModelIndex& idx)
{
	//too late...
}

void GraphsPlainModel::revert(const QModelIndex& idx)
{
	const QString& subgName = idx.data().toString();
	if (subgName.isEmpty())
	{
		//exitting new item
		removeRow(idx.row());
	}
}

void GraphsPlainModel::on_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{

}
	
void GraphsPlainModel::on_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last)
{

}
	
void GraphsPlainModel::on_rowsInserted(const QModelIndex& parent, int first, int last)
{

}
	
void GraphsPlainModel::on_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{

}
	
void GraphsPlainModel::on_rowsRemoved(const QModelIndex& parent, int first, int last)
{

}