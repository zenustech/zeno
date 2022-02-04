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

private:
	QStandardItem* appendSubModel(GraphsModel* pTreeModel, SubGraphModel* pModel);
};

#endif