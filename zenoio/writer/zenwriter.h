#ifndef __ZEN_WRITER_H__
#define __ZEN_WRITER_H__

#include <zeno/core/data.h>
#include "iocommon.h"

using namespace JsonHelper;

namespace zenoio
{
    class ZenWriter
    {
    public:
        ZenWriter();
        std::string dumpProgramStr(zeno::GraphData graph, AppSettings settings);
        void dumpToClipboard(const GraphData& nodes);

    private:
        void dumpNode(const NodeData& data, RAPIDJSON_WRITER& writer);
        void dumpSocket(zeno::ParamInfo info, RAPIDJSON_WRITER& writer);
        void _dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx, RAPIDJSON_WRITER& writer);
        void dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer);
        void dumpSettings(const AppSettings settings, RAPIDJSON_WRITER& writer);
    };
}

#endif