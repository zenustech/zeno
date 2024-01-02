#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>

namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_shared<Graph>())
{}

ZENO_API SubnetNode::~SubnetNode() = default;

void SubnetNode::init(const NodeData& dat)
{
    INode::init(dat);
    //需要检查SubInput/SubOutput是否对的上？
    if (dat.subgraph)
        subgraph->init(*dat.subgraph);
}

ZENO_API void SubnetNode::apply() {
    for (auto const &[key, nodeid]: subgraph->getSubInputs()) {
        auto subinput = safe_at(subgraph->nodes, nodeid, "node name").get();
        std::shared_ptr<IParam> spParam = get_input_param(key);
        if (spParam) {
            bool ret = subinput->set_output("port", spParam->result);
            assert(ret);
            ret = subinput->set_output("hasValue", std::make_shared<NumericObject>(true));
            assert(ret);
        }
        else {
            subinput->set_output("port", std::make_shared<DummyObject>());
            subinput->set_output("hasValue", std::make_shared<NumericObject>(false));
        }
    }

    std::set<std::string> nodesToExec;
    for (auto const &[key, nodeid]: subgraph->getSubOutputs()) {
        nodesToExec.insert(nodeid);
    }
    log_debug("{} subnet nodes to exec", nodesToExec.size());
    subgraph->applyNodes(nodesToExec);

    for (auto const &[key, nodeid]: subgraph->getSubOutputs()) {
        auto suboutput = safe_at(subgraph->nodes, nodeid, "node name").get();
        zany result = suboutput->get_input("port");
        if (result) {
            bool ret = set_output(key, result);
            assert(ret);
        }
    }
}

}
