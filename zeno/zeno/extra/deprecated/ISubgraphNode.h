#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>

namespace zeno {

struct ISubgraphNode : INode {
    virtual Graph *get_subgraph() = 0;
    ZENO_API virtual void apply() override;

    ZENO_API ISubgraphNode();
    ZENO_API virtual ~ISubgraphNode();
};

struct ISerialSubgraphNode : ISubgraphNode {
    std::unique_ptr<Graph> subg = nullptr;

    virtual const char *get_subgraph_json() = 0;
    ZENO_API virtual Graph *get_subgraph() override;

    ZENO_API ISerialSubgraphNode();
    ZENO_API virtual ~ISerialSubgraphNode();
};

struct SubgraphNode final : ISubgraphNode {
    std::unique_ptr<Graph> subgraph;
    std::unique_ptr<INodeClass> subgraphNodeClass;

    ZENO_API SubgraphNode();
    ZENO_API virtual ~SubgraphNode();

    ZENO_API virtual Graph *get_subgraph() override;
};

}
