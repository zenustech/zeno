#ifndef __ZEN_WRITER_H__
#define __ZEN_WRITER_H__

#include <zeno/core/data.h>
#include <zenoio/include/iocommon.h>

namespace zenoio
{
    class ZenWriter
    {
    public:
        ZenWriter();
        std::string dumpProgramStr(zeno::GraphData graph, AppSettings settings);
        std::string dumpToClipboard(const zeno::GraphData& nodes);

    private:
        void dumpGraph(zeno::GraphData graph, RAPIDJSON_WRITER& writer);
        void dumpNode(const zeno::NodeData& data, RAPIDJSON_WRITER& writer);
        void dumpSocket(zeno::ParamInfo info, RAPIDJSON_WRITER& writer);
        void dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer);
    };
}

#endif