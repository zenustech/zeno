#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>

namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_shared<Graph>(""))
{
    subgraph->optParentSubgNode = this;
}

ZENO_API SubnetNode::~SubnetNode() = default;

void SubnetNode::init(const NodeData& dat)
{
    //需要先初始化param
    //INode::init(dat);
    if (dat.name.empty())
        m_name = dat.name;

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
        sparam->m_wpNode = shared_from_this();
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
        sparam->name = m_name;
        sparam->type = param.type;
        sparam->m_wpNode = shared_from_this();
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
    sparam->m_wpNode = shared_from_this();
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
        inputs_.erase(name);
    }
    else {
        outputs_.erase(name);
    }
}

ZENO_API std::vector<std::shared_ptr<IParam>> SubnetNode::get_input_params() const
{
    std::vector<std::shared_ptr<IParam>> params;
    for (auto param : input_names) {
        auto it = inputs_.find(param);
        if (it == inputs_.end()) {
            zeno::log_warn("unknown param {}", param);
            continue;
        }
        params.push_back(it->second);
    }
    return params;
}

ZENO_API std::vector<std::shared_ptr<IParam>> SubnetNode::get_output_params() const
{
    std::vector<std::shared_ptr<IParam>> params;
    for (auto param : output_names) {
        auto it = outputs_.find(param);
        if (it == outputs_.end()) {
            zeno::log_warn("unknown param {}", param);
            continue;
        }
        params.push_back(it->second);
    }
    return params;
}

ZENO_API void SubnetNode::update_editparams(const std::vector<std::pair<zeno::ParamInfo, std::string>>& params)
{
    std::set<std::string> inputs_old, outputs_old;
    for (const auto& param_name : input_names) {
        inputs_old.insert(param_name);
    }
    for (const auto& param_name : output_names) {
        outputs_old.insert(param_name);
    }

    std::set<std::string> new_input_params, new_output_params;
    std::set<std::pair<std::string, std::string>> rename_input_params, rename_output_params;
    std::set<std::string> remove_input_params, remove_output_params;

    for (auto _pair : params) {
        const ParamInfo& param = _pair.first;
        const std::string oldname = _pair.second;
        const std::string newname = param.name;

        auto& in_outputs = param.bInput ? inputs_ : outputs_;
        auto& new_params = param.bInput ? new_input_params : new_output_params;
        auto& remove_params = param.bInput ? remove_input_params : remove_output_params;
        auto& rename_params = param.bInput ? rename_input_params : rename_output_params;

        if (oldname.empty()) {
            //new added name.
            if (in_outputs.find(newname) != in_outputs.end()) {
                // the new name happen to have the same name with the old name, but they are not the same param.
                in_outputs.erase(newname);
                if (param.bInput)
                    inputs_old.erase(newname);
                else
                    outputs_old.erase(newname);

                remove_params.insert(newname);
            }

            std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
            sparam->defl = param.defl;
            sparam->name = newname;
            sparam->type = param.type;
            sparam->control = param.control;
            sparam->m_wpNode = shared_from_this();
            in_outputs[newname] = sparam;

            new_params.insert(newname);
        }
        else if (in_outputs.find(oldname) != in_outputs.end()) {
            if (oldname != newname) {
                //exist name changed.
                in_outputs[newname] = in_outputs[oldname];
                in_outputs.erase(oldname);

                rename_params.insert({ oldname, newname });
            }
            else {
                //name stays.
            }

            if (param.bInput)
                inputs_old.erase(oldname);
            else
                outputs_old.erase(oldname);

            auto spParam = in_outputs[newname];
            spParam->type = param.type;
            spParam->defl = param.defl;
            spParam->name = newname;
            spParam->control = param.control;
        }
        else {
            throw makeError<KeyError>(oldname, "the name does not exist on the node");
        }
    }

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        inputs_.erase(rem_name);
        remove_input_params.insert(rem_name);
    }
    //update the names.
    input_names.clear();
    for (const auto& [param, _] : params) {
        if (param.bInput)
            input_names.push_back(param.name);
    }

    for (auto rem_name : outputs_old) {
        outputs_.erase(rem_name);
        remove_output_params.insert(rem_name);
    }
    output_names.clear();
    for (const auto& [param, _] : params) {
        if (!param.bInput)
            output_names.push_back(param.name);
    }

    //update subnetnode.
    for (auto name : new_input_params) {
        subgraph->createNode("SubInput", name);
    }
    for (const auto& [old_name, new_name] : rename_input_params) {
        subgraph->updateNodeName(old_name, new_name);
    }
    for (auto name : remove_input_params) {
        subgraph->removeNode(name);
    }

    for (auto name : new_output_params) {
        subgraph->createNode("SubOutput", name);
    }
    for (const auto& [old_name, new_name] : rename_output_params) {
        subgraph->updateNodeName(old_name, new_name);
    }
    for (auto name : remove_output_params) {
        subgraph->removeNode(name);
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
