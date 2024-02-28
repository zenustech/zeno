#ifndef __ZDA_WRITER_H__
#define __ZDA_WRITER_H__

#include <zeno/core/data.h>
#include <zeno/io/iocommon.h>
#include "commonwriter.h"
#include <zeno/core/Assets.h>

namespace zenoio
{
    class ZdaWriter : public CommonWriter
    {
    public:
        ZENO_API ZdaWriter();
        ZENO_API std::string dumpAsset(zeno::ZenoAsset asset);
    };
}

#endif