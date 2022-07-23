#include <zeno/extra/ISubgraphNode.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
//#include <zeno/utils/zeno_p.h>

namespace zeno {

ZENO_API ISubgraphNode::ISubgraphNode() = default;
ZENO_API ISubgraphNode::~ISubgraphNode() = default;
ZENO_API void ISubgraphNode::apply() {
    if (!grap) {
        grap = getThisSession()->createGraph();
        auto json = get_subgraph_json();
        grap->loadGraph(json);
        for (auto const &[key, nodename]: grap->subOutputNodes) {
            grap->nodesToExec.insert(nodename);
        }
    }
    Graph &gra = *grap;
    //ZENO_P(json);
        //ZENO_P(gra.subOutputNodes.size());
        //ZENO_P(gra.subInputNodes.size());
    for (auto const &[key, nodename]: gra.subInputNodes) {
        auto *node = gra.nodes.at(nodename).get();
        bool hasValue = has_input(key);
        //printf("$$$ %s %s\n", key.c_str(), typeid(*get_input(key)).name());
        node->inputs["_IN_hasValue"] = std::make_shared<NumericObject>(hasValue);
        if (hasValue)
            node->inputs["_IN_port"] = get_input(key);
        else
            node->inputs["_IN_port"] = std::make_shared<DummyObject>();
    }
    gra.applyNodesToExec();
    for (auto const &[key, nodename]: gra.subOutputNodes) {
        auto *node = gra.nodes.at(nodename).get();
        set_output(key, node->outputs.at("_OUT_port"));
    }
}

}
