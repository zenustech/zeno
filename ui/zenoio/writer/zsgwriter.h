#ifndef __ZSG_WRITER_H__
#define __ZSG_WRITER_H__

#include <zenoui/model/modeldata.h>
#include <zenoui/include/igraphsmodel.h>

class ZsgWriter
{
public:
	static ZsgWriter& getInstance();
	QString dumpProgramStr(IGraphsModel* pModel);
	QJsonObject dumpGraphs(IGraphsModel* pModel);
	QJsonObject dumpNode(const NODE_DATA& data);

private:
	ZsgWriter();
	QJsonObject _dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx);
	QJsonObject _dumpDescriptors(const NODE_DESCS& descs);
};

#endif