#ifndef __COMMON_WRITER_H__
#define __COMMON_WRITER_H__

#include <zeno/core/data.h>
#include <zeno/io/iocommon.h>

namespace zenoio
{
    class CommonWriter
    {
    public:
        CommonWriter();

    protected:
        void dumpGraph(zeno::GraphData graph, RAPIDJSON_WRITER& writer);
        void dumpNode(const zeno::NodeData& data, RAPIDJSON_WRITER& writer);
        void dumpSocket(zeno::ParamInfo info, RAPIDJSON_WRITER& writer);
        void dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer);
    };
}

#endif