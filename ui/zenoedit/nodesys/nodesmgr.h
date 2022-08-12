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
    static void initInputSocks(IGraphsModel* pModel, const QString& nodeid, INPUT_SOCKETS& descInputs);
	static void initOutputSocks(IGraphsModel* pModel, const QString& nodeid, OUTPUT_SOCKETS& descOutputs);
	static void initParams(const QString& descName, IGraphsModel* pModel, PARAMS_INFO& params);
	static PARAMS_INFO initParamsNotDesc(const QString& name);
};


#endif
