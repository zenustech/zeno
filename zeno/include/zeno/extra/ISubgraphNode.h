#pragma once

#include <zeno/zeno.h>

namespace zeno {

struct ISubgraphNode : zeno::INode {
    virtual zeno::Graph *get_subgraph() = 0;
    ZENO_API virtual void apply() override;

    ZENO_API ISubgraphNode();
    ZENO_API virtual ~ISubgraphNode();
};

struct ISerialSubgraphNode : ISubgraphNode {
    std::unique_ptr<zeno::Graph> subg = nullptr;

    virtual const char *get_subgraph_json() = 0;
    ZENO_API virtual zeno::Graph *get_subgraph() override;

    ZENO_API ISerialSubgraphNode();
    ZENO_API virtual ~ISerialSubgraphNode();
};

}
