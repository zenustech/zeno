#include <zeno/extra/ISubgraphNode.h>
#include <zeno/core/Graph.h>

namespace zeno {

ZENO_API ISubgraphNode::ISubgraphNode() = default;
ZENO_API ISubgraphNode::~ISubgraphNode() = default;
ZENO_API void ISubgraphNode::apply() {
    auto json = get_subgraph_json();
    Graph gra;
    gra.loadGraph(json);
    gra.subInputNodes.
    gra.applyNodesToExec();
}

}
