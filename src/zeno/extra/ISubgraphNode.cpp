#include <zeno/zeno.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/ISubgraphNode.h>

namespace zeno {

ZENO_API void ISubgraphNode::apply() {
    auto subg = get_subgraph();

    // VIEW subnodes only if subgraph is VIEW'ed
    subg->isViewed = has_option("VIEW");

    for (auto const &[key, obj]: inputs) {
        subg->setGraphInput(key, obj);
    }
    subg->applyGraph();

    for (auto &[key, obj]: subg->subOutputs) {
        if (subg->isViewed && !subg->hasAnyView) {
            getGlobalState()->addViewObject(obj);
            subg->hasAnyView = true;
        }
        set_output(key, std::move(obj));
    }

    subg->subInputs.clear();
    subg->subOutputs.clear();
}

ZENO_API ISubgraphNode::ISubgraphNode() = default;
ZENO_API ISubgraphNode::~ISubgraphNode() = default;

ZENO_API zeno::Graph *ISerialSubgraphNode::get_subgraph() {
    if (!subg) {
        subg = std::make_unique<zeno::Graph>();
        subg->scene = graph->scene;
        auto json = get_subgraph_json();
        subg->loadGraph(json);
    }
    return subg.get();
}

ZENO_API ISerialSubgraphNode::ISerialSubgraphNode() = default;
ZENO_API ISerialSubgraphNode::~ISerialSubgraphNode() = default;

ZENO_API Graph *SubgraphNode::get_subgraph() {
    return graph.get();
}

ZENO_API SubgraphNode::SubgraphNode() = default;
ZENO_API SubgraphNode::~SubgraphNode() = default;

ZENDEFNODE(SubgraphNode, {
    {},
    {},
    {},
    {"subgraph"},
});

}
