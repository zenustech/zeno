#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/types/DummyObject.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>
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
    INode::initParams(dat);
    //需要检查SubInput/SubOutput是否对的上？
    if (dat.subgraph && subgraph->getNodes().empty())
        subgraph->init(*dat.subgraph);
}

ZENO_API std::shared_ptr<Graph> SubnetNode::get_graph() const
{
    return subgraph;
}

ZENO_API bool SubnetNode::isAssetsNode() const {
    return subgraph->isAssets();
}

ZENO_API params_change_info SubnetNode::update_editparams(const ParamsUpdateInfo& params)
{
    //TODO: 这里只有primitive参数类型的情况，还需要整合obj参数的情况。
    std::set<std::string> inputs_old, outputs_old, obj_inputs_old, obj_outputs_old;
    for (const auto& param_name : m_input_names) {
        inputs_old.insert(param_name);
    }
    for (const auto& param_name : m_output_names) {
        outputs_old.insert(param_name);
    }
    for (const auto& param_name : m_obj_input_names) {
        obj_inputs_old.insert(param_name);
    }
    for (const auto& param_name : m_obj_output_names) {
        obj_outputs_old.insert(param_name);
    }

    params_change_info changes;

    for (auto _pair : params) {
        if (const auto& pParam = std::get_if<ParamObject>(&_pair.param))
        {
            const ParamObject& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& in_outputs = param.bInput ? m_inputObjs : m_outputObjs;
            auto& new_params = param.bInput ? changes.new_inputs : changes.new_outputs;
            auto& remove_params = param.bInput ? changes.remove_inputs : changes.remove_outputs;
            auto& rename_params = param.bInput ? changes.rename_inputs : changes.rename_outputs;

            if (oldname.empty()) {
                //new added name.
                if (in_outputs.find(newname) != in_outputs.end()) {
                    // the new name happen to have the same name with the old name, but they are not the same param.
                    in_outputs.erase(newname);
                    if (param.bInput)
                        obj_inputs_old.erase(newname);
                    else
                        obj_outputs_old.erase(newname);

                    remove_params.insert(newname);
                }

                std::unique_ptr<ObjectParam> sparam = std::make_unique<ObjectParam>();
                sparam->name = newname;
                sparam->type = param.type;
                sparam->socketType = param.socketType;
                sparam->m_wpNode = shared_from_this();
                in_outputs[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (in_outputs.find(oldname) != in_outputs.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    in_outputs[newname] = std::move(in_outputs[oldname]);
                    in_outputs.erase(oldname);

                    rename_params.insert({ oldname, newname });
                }
                else {
                    //name stays.
                }

                if (param.bInput)
                    obj_inputs_old.erase(oldname);
                else
                    obj_outputs_old.erase(oldname);

                auto& spParam = in_outputs[newname];
                spParam->type = param.type;
                spParam->name = newname;
                if (param.bInput)
                {
                    update_param_socket_type(spParam->name, param.socketType);
                }
            }
            else {
                throw makeError<KeyError>(oldname, "the name does not exist on the node");
            }
        }
        else if (const auto& pParam = std::get_if<ParamPrimitive>(&_pair.param))
        {
            const ParamPrimitive& param = *pParam;
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& in_outputs = param.bInput ? m_inputPrims : m_outputPrims;
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

                std::unique_ptr<PrimitiveParam> sparam = std::make_unique<PrimitiveParam>();
                sparam->defl = param.defl;
                sparam->name = newname;
                sparam->type = param.type;
                sparam->control = param.control;
                sparam->optCtrlprops = param.ctrlProps;
                sparam->socketType = param.socketType;
                sparam->m_wpNode = shared_from_this();
                in_outputs[newname] = std::move(sparam);

                new_params.insert(newname);
            }
            else if (in_outputs.find(oldname) != in_outputs.end()) {
                if (oldname != newname) {
                    //exist name changed.
                    in_outputs[newname] = std::move(in_outputs[oldname]);
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

                auto& spParam = in_outputs[newname];
                spParam->defl = param.defl;
                spParam->name = newname;
                spParam->socketType = param.socketType;
                if (param.bInput)
                {
                    update_param_type(spParam->name, param.type);
                    update_param_control(spParam->name, param.control);
                    update_param_control_prop(spParam->name, param.ctrlProps.value());
                }
            }
            else {
                throw makeError<KeyError>(oldname, "the name does not exist on the node");
            }
        }
    }

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        m_inputPrims.erase(rem_name);
        changes.remove_inputs.insert(rem_name);
    }

    for (auto rem_name : outputs_old) {
        m_outputPrims.erase(rem_name);
        changes.remove_outputs.insert(rem_name);
    }
    
    for (auto rem_name : obj_inputs_old) {
        m_inputObjs.erase(rem_name);
        changes.remove_inputs.insert(rem_name);
    }

    for (auto rem_name : obj_outputs_old) {
        m_outputObjs.erase(rem_name);
        changes.remove_outputs.insert(rem_name);
    }
    //update the names.
    m_input_names.clear();
    m_output_names.clear();
    m_obj_input_names.clear();
    m_obj_output_names.clear();
    changes.inputs.clear();
    changes.outputs.clear();
    for (const auto& [param, _] : params) {
        if (auto paramPrim = std::get_if<ParamPrimitive>(&param))
        {
            if (paramPrim->bInput)
            {
                m_input_names.push_back(paramPrim->name);
                changes.inputs.push_back(paramPrim->name);
            }
            else
            {
                m_output_names.push_back(paramPrim->name);
                changes.outputs.push_back(paramPrim->name);
            }
        }
        else if (auto paramPrim = std::get_if<ParamObject>(&param))
        {
            if (paramPrim->bInput)
            {
                m_obj_input_names.push_back(paramPrim->name);
                changes.inputs.push_back(paramPrim->name);
            }
            else
            {
                m_obj_output_names.push_back(paramPrim->name);
                changes.outputs.push_back(paramPrim->name);
            }
        }
    }
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

void SubnetNode::mark_subnetdirty(bool bOn)
{
    if (bOn) {
        subgraph->markDirtyAll();
    }
}

ZENO_API void SubnetNode::apply() {
    for (auto const &subinput_node: subgraph->getSubInputs()) {
        auto subinput = subgraph->getNode(subinput_node);
        auto iter = m_inputObjs.find(subinput_node);
        if (iter != m_inputObjs.end()) {
            //object type.
            bool ret = subinput->set_output("port", iter->second->spObject);
            assert(ret);
            ret = subinput->set_output("hasValue", std::make_shared<NumericObject>(true));
            assert(ret);
        }
        else {
            //primitive type
            auto iter2 = m_inputPrims.find(subinput_node);
            if (iter2 != m_inputPrims.end()) {
                bool ret = subinput->set_primitive_output("port", iter2->second->result);
                assert(ret);
                ret = subinput->set_output("hasValue", std::make_shared<NumericObject>(true));
                assert(ret);
            }
            else {
                subinput->set_output("port", std::make_shared<DummyObject>());
                subinput->set_output("hasValue", std::make_shared<NumericObject>(false));
            }
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
    }
    node.customUi = m_customUi;
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
