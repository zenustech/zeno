#ifndef __ZEN_WRITER_H__
#define __ZEN_WRITER_H__

#include <zeno/core/data.h>
#include <zeno/io/iocommon.h>
#include "commonwriter.h"

namespace zenoio
{
    class ZenWriter : public CommonWriter
    {
    public:
        ZENO_API ZenWriter();
        ZENO_API std::string dumpProgramStr(zeno::GraphData graph, AppSettings settings);
        ZENO_API std::string dumpToClipboard(const zeno::NodesData& nodes);
    };
}

#endif