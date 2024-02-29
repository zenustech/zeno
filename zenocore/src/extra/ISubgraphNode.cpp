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
    /*
    if (!grap) {
        grap = getThisSession()->createGraph("");
        auto json = get_subgraph_json();
        grap->loadGraph(json);
        for (auto const &[key, nodename]: grap->getSubOutputs()) {
            grap->nodesToExec.insert(nodename);
        }
    }
    Graph &gra = *grap;

    for (auto const &[key, nodename]: gra.getSubInputs()) {
        auto *node = gra.m_nodes.at(nodename).get();
        bool hasValue = has_input(key);

        
        node->inputs["_IN_hasValue"] = std::make_shared<NumericObject>(hasValue);
        if (hasValue)
            node->inputs["_IN_port"] = get_input(key);
        else
            node->inputs["_IN_port"] = std::make_shared<DummyObject>();
        
    }
    gra.runGraph();
    for (auto const &[key, nodename]: gra.getSubOutputs()) {
        auto *node = gra.m_nodes.at(nodename).get();
        //set_output(key, node->outputs.at("_OUT_port"));
    }
    */
}

}
