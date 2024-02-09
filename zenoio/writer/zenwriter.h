#ifndef __ZEN_WRITER_H__
#define __ZEN_WRITER_H__

#include <zeno/core/data.h>
#include <zenoio/include/iocommon.h>
#include "commonwriter.h"

namespace zenoio
{
    class ZenWriter : public CommonWriter
    {
    public:
        ZenWriter();
        std::string dumpProgramStr(zeno::GraphData graph, AppSettings settings);
        std::string dumpToClipboard(const zeno::GraphData& nodes);
    };
}

#endif