#ifndef __NODES_MGR_H__
#define __NODES_MGR_H__

#include <zenoui/include/igraphsmodel.h>
#include <zenoui/model/modeldata.h>
#include <zenoui/model/modelrole.h>
#include <QtWidgets>

class NodesMgr : public QObject
{
	Q_OBJECT
public:
	static void createNewNode(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, const QPointF& pt);
	static NODE_TYPE nodeType(const QString& name);
	static QList<QAction*> getCategoryActions(IGraphsModel* pModel, QModelIndex subgIdx, const QString& filter, QPointF scenePos);
};


#endif