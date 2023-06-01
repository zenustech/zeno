#ifndef __IO_COMMON_H__
#define __IO_COMMON_H__

#include <zenomodel/include/modeldata.h>

namespace zenoio {
    enum ZSG_VERSION
    {
        VER_2,          //old version io
        VER_2_5,        //new version io
        VER_3,          //the lastest io format, supporting tree layout.
    };

    struct ZSG_PARSE_RESULT
    {
        QMap<QString, SUBGRAPH_DATA> subgraphs;
        SUBGRAPH_DATA mainGraph;
        ZSG_VERSION ver;
        NODE_DESCS descs;
        TIMELINE_INFO timeline;
    };
}

#endif