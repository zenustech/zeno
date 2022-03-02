#ifndef __GRAPHICS_TREEMODEL_H__
#define __GRAPHICS_TREEMODEL_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;
class IGraphsModel;

class GraphsTreeModel : public QStandardItemModel
{
	Q_OBJECT
public:
	GraphsTreeModel(QObject* parent = nullptr);
	~GraphsTreeModel();
	void init(IGraphsModel* pModel);

public slots:
	void on_graphs_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_subg_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
	void on_subg_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void on_subg_rowsInserted(const QModelIndex& parent, int first, int last);
	void on_subg_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_subg_rowsRemoved(const QModelIndex& parent, int first, int last);

private:
	QStandardItem* appendSubModel(SubGraphModel* pModel);

	GraphsModel* m_model;
};

#endif