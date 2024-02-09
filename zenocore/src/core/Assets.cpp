#include <zeno/core/Assets.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/core/IParam.h>


namespace zeno {

ZENO_API AssetsMgr::AssetsMgr() {

}

ZENO_API AssetsMgr::~AssetsMgr() {

}

ZENO_API void AssetsMgr::createAsset(const zeno::AssetInfo asset) {
    Asset newAsst;

    newAsst.m_info = asset;

    std::shared_ptr<Graph> spGraph = std::make_shared<Graph>(asset.name, true);

    spGraph->setName(asset.name);
    //manually init some args nodes.
    spGraph->createNode("SubInput", "input1", "", { 0, 0 });
    spGraph->createNode("SubInput", "input2", "", { 0,700 });
    spGraph->createNode("SubOutput", "output1", "", { 1300, 250 });
    spGraph->createNode("SubOutput", "output2", "", { 1300, 900 });

    ParamInfo param;
    param.name = "input1";
    param.bInput = true;
    newAsst.inputs.push_back(param);

    param.name = "input2";
    param.bInput = true;
    newAsst.inputs.push_back(param);

    param.name = "output1";
    param.bInput = false;
    newAsst.outputs.push_back(param);

    param.name = "output2";
    param.bInput = false;
    newAsst.outputs.push_back(param);

    newAsst.sharedGraph = spGraph;

    m_assets.insert(std::make_pair(asset.name, newAsst));
    CALLBACK_NOTIFY(createAsset, asset)
}

ZENO_API void AssetsMgr::removeAsset(const std::string& name) {
    m_assets.erase(name);
    CALLBACK_NOTIFY(removeAsset, name)
}

ZENO_API void AssetsMgr::renameAsset(const std::string& old_name, const std::string& new_name) {
    //TODO
    CALLBACK_NOTIFY(renameAsset, old_name, new_name)
}

ZENO_API Asset AssetsMgr::getAsset(const std::string& name) const {
    if (m_assets.find(name) != m_assets.end()) {
        return m_assets.at(name);
    }
    return Asset();
}

ZENO_API void AssetsMgr::updateAssets(const std::string name, ParamsUpdateInfo info) {
    if (m_assets.find(name) == m_assets.end()) {
        return;
    }
    auto& assets = m_assets[name];
    std::set<std::string> inputs_old, outputs_old;

    std::set<std::string> input_names;
    std::set<std::string> output_names;
    for (auto param : assets.inputs) {
        input_names.insert(param.name);
    }
    for (auto param : assets.outputs) {
        output_names.insert(param.name);
    }

    for (const auto& param_name : input_names) {
        inputs_old.insert(param_name);
    }
    for (const auto& param_name : output_names) {
        outputs_old.insert(param_name);
    }

    params_change_info changes;

    for (auto _pair : info) {
        const ParamInfo& param = _pair.param;
        const std::string oldname = _pair.oldName;
        const std::string newname = param.name;

        auto& in_outputs = param.bInput ? input_names : output_names;
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
            new_params.insert(newname);
        }
        else if (in_outputs.find(oldname) != in_outputs.end()) {
            if (oldname != newname) {
                //exist name changed.
                in_outputs.insert(newname);
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
        }
        else {
            throw makeError<KeyError>(oldname, "the name does not exist on the node");
        }
    }

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        changes.remove_inputs.insert(rem_name);
    }
    //update the names.
    input_names.clear();
    for (const auto& [param, _] : info) {
        if (param.bInput)
            input_names.insert(param.name);
    }

    for (auto rem_name : outputs_old) {
        changes.remove_outputs.insert(rem_name);
    }
    output_names.clear();
    for (const auto& [param, _] : info) {
        if (!param.bInput)
            output_names.insert(param.name);
    }

    //update subnetnode.
    for (auto name : changes.new_inputs) {
        assets.sharedGraph->createNode("SubInput", name);
    }
    for (const auto& [old_name, new_name] : changes.rename_inputs) {
        assets.sharedGraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_inputs) {
        assets.sharedGraph->removeNode(name);
    }

    for (auto name : changes.new_outputs) {
        assets.sharedGraph->createNode("SubOutput", name);
    }
    for (const auto& [old_name, new_name] : changes.rename_outputs) {
        assets.sharedGraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_outputs) {
        assets.sharedGraph->removeNode(name);
    }

    //update assets data
    assets.inputs.clear();
    assets.outputs.clear();
    for (auto pair : info) {
        if (pair.param.bInput)
            assets.inputs.push_back(pair.param);
        else
            assets.outputs.push_back(pair.param);
    }
}

ZENO_API std::shared_ptr<INode> AssetsMgr::newInstance(const std::string& assetsName, const std::string& nodeName) {
    if (m_assets.find(assetsName) == m_assets.end()) {
        return nullptr;
    }

    auto& assets = m_assets[assetsName];

    std::shared_ptr<SubnetNode> spNode = std::make_shared<SubnetNode>();
    spNode->subgraph = assets.sharedGraph;
    spNode->m_nodecls = assetsName;
    spNode->m_name = nodeName;

    for (const ParamInfo& param : assets.inputs)
    {
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->m_wpNode = spNode;
        spNode->add_input_param(sparam);
        spNode->m_input_names.push_back(param.name);
    }

    for (const ParamInfo& param : assets.outputs)
    {
        std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
        sparam->defl = param.defl;
        sparam->name = param.name;
        sparam->type = param.type;
        sparam->m_wpNode = spNode;
        spNode->add_output_param(sparam);
        spNode->m_output_names.push_back(param.name);
    }

    return std::dynamic_pointer_cast<INode>(spNode);
}

}