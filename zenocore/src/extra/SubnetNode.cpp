#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>
#include <zeno/core/Assets.h>

namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_shared<Graph>(""))
{
    subgraph->optParentSubgNode = this;

    auto cl = safe_at(getSession().nodeClasses, "Subnet", "node class name").get();
    m_customUi = cl->m_customui;
}

ZENO_API SubnetNode::~SubnetNode() = default;

ZENO_API void SubnetNode::initParams(const NodeData& dat)
{
    for (const ParamInfo& param : dat.inputs)
    {
        if (m_inputs.find(param.name) != m_inputs.end())
            continue;
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->socketType = param.socketType;
        sparam->control = param.control;
        sparam->optCtrlprops = param.ctrlProps;
        sparam->m_wpNode = shared_from_this();
        add_input_param(sparam);
        m_input_names.push_back(param.name);
    }

    for (const ParamInfo& param : dat.outputs)
    {
        if (m_outputs.find(param.name) != m_outputs.end())
            continue;
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->socketType = PrimarySocket;
        sparam->m_wpNode = shared_from_this();
        add_output_param(sparam);
        m_output_names.push_back(param.name);
    }

    //需要检查SubInput/SubOutput是否对的上？
    if (dat.subgraph && subgraph->getNodes().empty())
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
        m_inputs.erase(name);
    }
    else {
        m_outputs.erase(name);
    }
}

ZENO_API std::shared_ptr<Graph> SubnetNode::get_graph() const
{
    return subgraph;
}

ZENO_API bool SubnetNode::isAssetsNode() const {
    return subgraph->isAssets();
}

ZENO_API std::vector<std::shared_ptr<IParam>> SubnetNode::get_input_params() const
{
    std::vector<std::shared_ptr<IParam>> params;
    for (auto param : m_input_names) {
        auto it = m_inputs.find(param);
        if (it == m_inputs.end()) {
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
    for (auto param : m_output_names) {
        auto it = m_outputs.find(param);
        if (it == m_outputs.end()) {
            zeno::log_warn("unknown param {}", param);
            continue;
        }
        params.push_back(it->second);
    }
    return params;
}

ZENO_API params_change_info SubnetNode::update_editparams(const ParamsUpdateInfo& params)
{
    std::set<std::string> inputs_old, outputs_old;
    for (const auto& param_name : m_input_names) {
        inputs_old.insert(param_name);
    }
    for (const auto& param_name : m_output_names) {
        outputs_old.insert(param_name);
    }

    params_change_info changes;

    for (auto _pair : params) {
        const ParamInfo& param = _pair.param;
        const std::string oldname = _pair.oldName;
        const std::string newname = param.name;

        auto& in_outputs = param.bInput ? m_inputs : m_outputs;
        auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
        auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
        auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

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
            sparam->optCtrlprops = param.ctrlProps;
            sparam->socketType = param.socketType;
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
            spParam->optCtrlprops = param.ctrlProps;
            spParam->socketType = param.socketType;
        }
        else {
            throw makeError<KeyError>(oldname, "the name does not exist on the node");
        }
    }

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        m_inputs.erase(rem_name);
        changes.remove_inputs.insert(rem_name);
    }
    //update the names.
    m_input_names.clear();
    for (const auto& [param, _] : params) {
        if (param.bInput)
            m_input_names.push_back(param.name);
    }
    changes.inputs = m_input_names;

    for (auto rem_name : outputs_old) {
        m_outputs.erase(rem_name);
        changes.remove_outputs.insert(rem_name);
    }
    m_output_names.clear();
    for (const auto& [param, _] : params) {
        if (!param.bInput)
            m_output_names.push_back(param.name);
    }
    changes.outputs = m_output_names;

    //update subnetnode.
    if (!subgraph->isAssets()) {
        for (auto name : changes.new_inputs) {
            subgraph->createNode("SubInput", name);
        }
        for (const auto& [old_name, new_name] : changes.rename_inputs) {
            subgraph->updateNodeName(old_name, new_name);
        }
        for (auto name : changes.remove_inputs) {
            subgraph->removeNode(name);
        }

        for (auto name : changes.new_outputs) {
            subgraph->createNode("SubOutput", name);
        }
        for (const auto& [old_name, new_name] : changes.rename_outputs) {
            subgraph->updateNodeName(old_name, new_name);
        }
        for (auto name : changes.remove_outputs) {
            subgraph->removeNode(name);
        }
    }
    return changes;
}

ZENO_API void SubnetNode::apply() {
    for (auto const &subinput_node: subgraph->getSubInputs()) {
        auto subinput = subgraph->getNode(subinput_node);
        std::shared_ptr<IParam> spParam = get_input_param(subinput_node);
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
    for (auto const &suboutput_node: subgraph->getSubOutputs()) {
        nodesToExec.insert(suboutput_node);
    }
    subgraph->applyNodes(nodesToExec);

    for (auto const &suboutput_node: subgraph->getSubOutputs()) {
        auto suboutput = subgraph->getNode(suboutput_node);
        zany result = suboutput->get_input("port");
        if (result) {
            bool ret = set_output(suboutput_node, result);
            assert(ret);
        }
    }
}

ZENO_API NodeData SubnetNode::exportInfo() const {
    NodeData node = INode::exportInfo();
    Asset asset = zeno::getSession().assets->getAsset(node.cls);
    if (!asset.m_info.name.empty()) {
        node.asset = asset.m_info;
        node.type = Node_AssetInstance;
    }
    else {
        node.subgraph = subgraph->exportGraph();
        node.type = Node_SubgraphNode;

        node.customUi = m_customUi;
    }
    return node;
}

ZENO_API CustomUI SubnetNode::get_customui() const
{
    return m_customUi;
}

ZENO_API void SubnetNode::setCustomUi(const CustomUI& ui)
{
    m_customUi = ui;
}

}
