#ifndef __ZDA_WRITER_H__
#define __ZDA_WRITER_H__

#include <zeno/core/data.h>
#include <zenoio/include/iocommon.h>
#include "commonwriter.h"
#include <zeno/core/Assets.h>

namespace zenoio
{
    class ZdaWriter : public CommonWriter
    {
    public:
        ZdaWriter();
        std::string dumpAsset(zeno::ZenoAsset asset);
    };
}

#endif