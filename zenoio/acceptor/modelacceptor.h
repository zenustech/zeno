#ifndef __MODEL_ACCEPTOR_H__
#define __MODEL_ACCEPTOR_H__

#include "iacceptor.h"

class SubGraphModel;
class GraphsModel;

class ModelAcceptor : public IAcceptor
{
public:
	ModelAcceptor(GraphsModel* pModel);

	//IAcceptor
	void setDescriptors(const NODE_DESCS& nodesParams) override;
	void BeginSubgraph(const QString& name) override;
	void EndSubgraph() override;
	void setFilePath(const QString& fileName) override;
	void switchSubGraph(const QString& graphName) override;
	void addNode(const QString& nodeid, const QString& name, const NODE_DESCS& descriptors) override;
	void setViewRect(const QRectF& rc) override;
	void setSocketKeys(const QString& id, const QStringList& keys) override;
	void initSockets(const QString& id, const QString& name, const NODE_DESCS& descs) override;
	void setInputSocket(const QString& id, const QString& inSock, const QString& outId, const QString& outSock, const QVariant& defaultValue) override;
	void setParamValue(const QString& id, const QString& name, const QVariant& var) override;
	void setPos(const QString& id, const QPointF& pos) override;
	void setOptions(const QString& id, const QStringList& options) override;
	void setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps) override;
	void setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard) override;

private:
	void _initSockets(const QString& id, const QString& name, INPUT_SOCKETS& inputs, PARAMS_INFO& params, OUTPUT_SOCKETS& outputs);

	SubGraphModel* m_currentGraph;
	GraphsModel* m_pModel;
};


#endif