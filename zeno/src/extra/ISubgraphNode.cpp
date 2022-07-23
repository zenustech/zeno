#include <zeno/extra/ISubgraphNode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/utils/zeno_p.h>

namespace zeno {

ZENO_API ISubgraphNode::ISubgraphNode() = default;
ZENO_API ISubgraphNode::~ISubgraphNode() = default;
ZENO_API void ISubgraphNode::apply() {
    auto grap = getThisSession()->createGraph();
    Graph &gra = *grap;
    auto json = get_subgraph_json();
    gra.loadGraph(json);
    for (auto const &[key, nodename]: gra.subOutputNodes) {
        //gra.nodes.at(nodename);
        gra.nodesToExec.insert(nodename);
    }
    for (auto const &[key, nodename]: gra.subInputNodes) {
        auto *node = gra.nodes.at(nodename).get();
        auto hasValue = has_input(key);
        node->inputs["_IN_hasValue"] = objectFromLiterial(hasValue);
        if (hasValue)
            node->inputs["_IN_port"] = get_input(key);
    }
    gra.applyNodesToExec();
    for (auto const &[key, nodename]: gra.subOutputNodes) {
        auto *node = gra.nodes.at(nodename).get();
        set_output(key, node->outputs.at("_OUT_port"));
    }
}

}
