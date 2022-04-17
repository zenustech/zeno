#ifndef __ZSG_WRITER_H__
#define __ZSG_WRITER_H__

#include <zenoui/model/modeldata.h>
#include <zenoui/include/igraphsmodel.h>
#include <zenoui/util/jsonhelper.h>

using namespace JsonHelper;

class ZsgWriter
{
public:
	static ZsgWriter& getInstance();
	QString dumpProgramStr(IGraphsModel* pModel);

private:
	ZsgWriter();
	void dumpNode(const NODE_DATA& data, RAPIDJSON_WRITER& writer);
	void _dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx, RAPIDJSON_WRITER& writer);
	void _dumpDescriptors(const NODE_DESCS& descs, RAPIDJSON_WRITER& writer);
};

#endif