#ifndef __GRAPHICS_TREEMODEL_H__
#define __GRAPHICS_TREEMODEL_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;

class GraphsTreeModel : public QStandardItemModel
{
	Q_OBJECT
public:
	GraphsTreeModel(GraphsModel* pTreeModel, QObject* parent = nullptr);
	~GraphsTreeModel();
	void init(GraphsModel* pModel);

public slots:
	void on_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
	void on_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void on_rowsInserted(const QModelIndex& parent, int first, int last);
	void on_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_rowsRemoved(const QModelIndex& parent, int first, int last);

private:
	QStandardItem* appendSubModel(SubGraphModel* pModel);

	GraphsModel* m_model;
};

#endif