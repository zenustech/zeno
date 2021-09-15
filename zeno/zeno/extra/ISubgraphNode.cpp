#include <zeno/zeno.h>
#ifdef ZENO_VISUALIZATION
#include <zeno/extra/Visualization.h>
#endif
#include <zeno/extra/ISubgraphNode.h>

namespace zeno {

void ISubgraphNode::apply() {
    auto subg = get_subgraph();

#ifdef ZENO_VISUALIZATION
    // VIEW subnodes only if subgraph is VIEW'ed
    subg->isViewed = has_option("VIEW");
#endif

    for (auto const &[key, obj]: inputs) {
        subg->setGraphInput2(key, obj);
    }
    subg->applyGraph();

    for (auto &[key, obj]: subg->subOutputs) {
#ifdef ZENO_VISUALIZATION
        if (subg->isViewed && !subg->hasAnyView) {
            auto path = zeno::Visualization::exportPath();
            if (auto p = zeno::silent_any_cast<
                    std::shared_ptr<zeno::IObject>>(obj); p.has_value()) {
                if (auto node = graph->getOverloadNode("ToVisualize", {p.value()}); node) {
                    node->inputs["path:"] = path;
                    node->doApply();
                }
            }
            subg->hasAnyView = true;
        }
#endif
        set_output2(key, std::move(obj));
    }

    subg->subInputs.clear();
    subg->subOutputs.clear();
}

ISubgraphNode::ISubgraphNode() = default;
ISubgraphNode::~ISubgraphNode() = default;

zeno::Graph *ISerialSubgraphNode::get_subgraph() {
    if (!subg) {
        subg = std::make_unique<zeno::Graph>();
        subg->scene = graph->scene;
        auto json = get_subgraph_json();
        subg->loadGraph(json);
    }
    return subg.get();
}

ISerialSubgraphNode::ISerialSubgraphNode() = default;
ISerialSubgraphNode::~ISerialSubgraphNode() = default;

}
