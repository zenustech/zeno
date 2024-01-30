#ifndef __IO_COMMON_H__
#define __IO_COMMON_H__

#include <zeno/core/data.h>

namespace zenoio {
    //already declared in zeno/core/common.h
    //enum ZSG_VERSION
    //{
    //    VER_2,          //old version io
    //    VER_2_5,        //new version io
    //};

    struct ZSG_PARSE_RESULT {
        zeno::GraphData mainGraph;
        zeno::ZSG_VERSION iover;
        zeno::NodeDescs descs;
        std::map<std::string, zeno::GraphData> sharedGraphs;
        zeno::TimelineInfo timeline;
    };
}

#endif