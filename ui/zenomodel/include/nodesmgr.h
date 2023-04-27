#ifndef __NODES_MGR_H__
#define __NODES_MGR_H__

#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>
#include <QtWidgets>

class NodesMgr : public QObject
{
	Q_OBJECT
public:
	static QString createNewNode(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, const QPointF& pt);
	static NODE_DATA newNodeData(IGraphsModel* pModel, const QString &descName, const QPointF& pt = QPointF(0, 0));
	static NODE_TYPE nodeType(const QString& name);
    static void initInputSocks(IGraphsModel* pModel, const QString& nodeid, INPUT_SOCKETS& descInputs, bool isSubgraph);
	static void initOutputSocks(IGraphsModel* pModel, const QString& nodeid, OUTPUT_SOCKETS& descOutputs);
	static void initParams(const QString& descName, IGraphsModel* pModel, PARAMS_INFO& params);
	static PARAMS_INFO initParamsNotDesc(const QString& name);
};


#endif
