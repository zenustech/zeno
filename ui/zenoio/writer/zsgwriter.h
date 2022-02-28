#ifndef __ZSG_WRITER_H__
#define __ZSG_WRITER_H__

#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modeldata.h>
#include <zenoui/model/subgraphmodel.h>

class ZsgWriter
{
public:
	static ZsgWriter& getInstance();
	QString dumpProgramStr(GraphsModel* pModel);
	QString dumpSubGraph(SubGraphModel* pSubModel);
	QJsonObject dumpGraphs(GraphsModel* pMode);
	QJsonObject dumpNode(const NODE_DATA& data);

private:
	ZsgWriter();
	QJsonObject _dumpSubGraph(SubGraphModel* pSubModel);
	QJsonObject _dumpDescriptors(const NODE_DESCS& descs);
};

#endif