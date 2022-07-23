#pragma once

#include <zeno/core/INode.h>

namespace zeno {

struct Graph;

struct ISubgraphNode : INode {
    virtual const char *get_subgraph_json() = 0;
    std::unique_ptr<Graph> grap;

    ZENO_API ISubgraphNode();
    ZENO_API virtual ~ISubgraphNode() override;
    ZENO_API virtual void apply() override;
};

using ISerialSubgraphNode = ISubgraphNode;

}
