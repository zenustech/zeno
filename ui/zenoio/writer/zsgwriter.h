#ifndef __ZSG_WRITER_H__
#define __ZSG_WRITER_H__

#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/jsonhelper.h>
#include "common.h"

//#define LEGACY_ZSG_WRITE

using namespace JsonHelper;

class ZsgWriter
{
public:
    static ZsgWriter& getInstance();
    QString dumpProgramStr(IGraphsModel* pModel, APP_SETTINGS settings);
    void dumpToClipboard(const QMap<QString, NODE_DATA>& nodes);

private:
    ZsgWriter();
    void dumpNode(const NODE_DATA& data, RAPIDJSON_WRITER& writer);
    void dumpSocket(SOCKET_INFO info, bool bInput, RAPIDJSON_WRITER& writer, bool isDesc);
    void _dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx, RAPIDJSON_WRITER& writer);
    void _dumpDescriptors(const NODE_DESCS& descs, RAPIDJSON_WRITER& writer);
    void dumpTimeline(TIMELINE_INFO info, RAPIDJSON_WRITER& writer);
    void dumpParams(const PARAM_INFO &info, RAPIDJSON_WRITER &writer);
    void dumpSubGraphDesc(const NODE_DESC &desc, RAPIDJSON_WRITER &writer);
};

#endif