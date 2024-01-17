#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>

namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_shared<Graph>(""))
{}

ZENO_API SubnetNode::~SubnetNode() = default;

void SubnetNode::init(const NodeData& dat)
{
    //需要先初始化param
    //INode::init(dat);
    if (dat.name.empty())
        name = dat.name;

    for (const ParamInfo& param : dat.inputs)
    {
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        if (!sparam) {
            zeno::log_warn("input param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->m_spNode = shared_from_this();
        add_input_param(sparam);
    }

    for (const ParamInfo& param : dat.outputs)
    {
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        if (!sparam) {
            zeno::log_warn("output param `{}` is not registerd in current zeno version");
            continue;
        }
        sparam->defl = param.defl;
        sparam->name = name;
        sparam->type = param.type;
        sparam->m_spNode = shared_from_this();
        add_output_param(sparam);
    }

    //需要检查SubInput/SubOutput是否对的上？
    if (dat.subgraph)
        subgraph->init(*dat.subgraph);
}

ZENO_API void SubnetNode::add_param(bool bInput, const ParamInfo& param)
{
    std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
    sparam->name = param.name;
    sparam->m_spNode = shared_from_this();
    sparam->type = param.type;
    sparam->defl = param.defl;
    if (bInput) {
        add_input_param(sparam);
    }
    else {
        add_output_param(sparam);
    }
}

ZENO_API void SubnetNode::remove_param(bool bInput, const std::string& name)
{
    if (bInput) {
        for (int i = 0; i < inputs_.size(); i++) {
            if (inputs_[i]->name == name) {
                inputs_.erase(inputs_.begin() + i);
                break;
            }
        }
    }
    else {
        for (int i = 0; i < outputs_.size(); i++) {
            if (outputs_[i]->name == name) {
                outputs_.erase(outputs_.begin() + i);
                break;
            }
        }
    }
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
