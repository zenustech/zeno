#include <zeno/core/Assets.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/core/CoreParam.h>
#include <filesystem>
#include <zeno/io/zdareader.h>
#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif
#include <zeno/core/typeinfo.h>


namespace zeno {

ZENO_API AssetsMgr::AssetsMgr() {
    initAssetsInfo();
}

ZENO_API AssetsMgr::~AssetsMgr() {

}

void AssetsMgr::initAssetsInfo() {
#ifdef _WIN32
    WCHAR documents[MAX_PATH];
    SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, documents);

    std::filesystem::path docPath(documents);

    std::filesystem::path zenoDir = std::filesystem::u8path(docPath.string() + "/Zeno");
    if (!std::filesystem::is_directory(zenoDir)) {
        std::filesystem::create_directories(zenoDir);
    }
    std::filesystem::path assetsDir = std::filesystem::u8path(zenoDir.string() + "/assets");
    if (!std::filesystem::is_directory(assetsDir)) {
        std::filesystem::create_directories(assetsDir);
    }

    for (auto const& dir_entry : std::filesystem::directory_iterator(assetsDir))
    {
        std::filesystem::path itemPath = dir_entry.path();
        if (itemPath.extension() == ".zda") {
            std::string zdaPath = itemPath.string();
            zenoio::ZdaReader reader;
            reader.setDelayReadGraph(true);
            zeno::scope_exit sp([&] {reader.setDelayReadGraph(false); });

            zenoio::ZSG_PARSE_RESULT result = reader.openFile(zdaPath);
            if (result.code == zenoio::PARSE_NOERROR) {
                zeno::ZenoAsset zasset = reader.getParsedAsset();
                zasset.info.path = zdaPath;
                createAsset(zasset);
            }
        }
    }
#endif
}

ZENO_API std::shared_ptr<Graph> AssetsMgr::getAssetGraph(const std::string& name, bool bLoadIfNotExist) {
    if (m_assets.find(name) != m_assets.end()) {
        if (!m_assets[name].sharedGraph) {
            zenoio::ZdaReader reader;
            reader.setDelayReadGraph(false);
            const AssetInfo& info = m_assets[name].m_info;
            std::string zdaPath = info.path;
            zenoio::ZSG_PARSE_RESULT result = reader.openFile(zdaPath);
            if (result.code == zenoio::PARSE_NOERROR) {
                zeno::ZenoAsset zasset = reader.getParsedAsset();
                assert(zasset.optGraph.has_value());
                std::shared_ptr<Graph> spGraph = std::make_shared<Graph>(info.name, true);
                spGraph->setName(info.name);
                spGraph->init(zasset.optGraph.value());
                m_assets[name].sharedGraph = spGraph;
            }
        }
        return m_assets[name].sharedGraph;
    }
    return nullptr;
}

ZENO_API void AssetsMgr::createAsset(const zeno::ZenoAsset asset, bool isFirstCreate) {
    Asset newAsst;

    newAsst.m_info = asset.info;
    if (asset.optGraph.has_value())
    {
        std::shared_ptr<Graph> spGraph = std::make_shared<Graph>(asset.info.name, true);
        spGraph->setName(asset.info.name);
        spGraph->init(asset.optGraph.value());
        newAsst.sharedGraph = spGraph;
    }
    newAsst.primitive_inputs = asset.primitive_inputs;
    newAsst.primitive_outputs = asset.primitive_outputs;
    newAsst.object_inputs = asset.object_inputs;
    newAsst.object_outputs = asset.object_outputs;
    newAsst.m_customui = asset.m_customui;

    if (isFirstCreate && asset.optGraph.has_value()) {
        initAssetSubInputOutput(newAsst);
    }
    if (m_assets.find(asset.info.name) != m_assets.end()) {
        m_assets[asset.info.name] = newAsst;
    }
    else {
        m_assets.insert(std::make_pair(asset.info.name, newAsst));
    }

    CALLBACK_NOTIFY(createAsset, asset.info)
}

ZENO_API void AssetsMgr::removeAsset(const std::string& name, bool deleteAssetFile) {
    m_assets.erase(name);
    if (deleteAssetFile) {
    #ifdef _WIN32
        WCHAR documents[MAX_PATH];
        SHGetFolderPathW(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, documents);
        std::filesystem::path docPath(documents);
        std::filesystem::path zenoDir = std::filesystem::u8path(docPath.string() + "/Zeno");
        if (std::filesystem::is_directory(zenoDir)) {
            std::filesystem::path assetsDir = std::filesystem::u8path(zenoDir.string() + "/assets");
            if (std::filesystem::is_directory(assetsDir)) {
                std::filesystem::path assetsfile = std::filesystem::u8path(assetsDir.string() + "/" + name + ".zda");
                if (std::filesystem::exists(assetsfile) && std::filesystem::is_regular_file(assetsfile)) {
                    std::filesystem::remove(assetsfile);
                }
            }
        }
    #endif
    }
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

ZENO_API std::vector<Asset> AssetsMgr::getAssets() const {
    std::vector<Asset> assets;
    for (auto& [name, asset] : m_assets) {
        assets.push_back(asset);
    }
    return assets;
}

ZENO_API void AssetsMgr::updateAssets(const std::string name, ParamsUpdateInfo info, const zeno::CustomUI& customui)
{
    if (m_assets.find(name) == m_assets.end()) {
        return;
    }
    auto& assets = m_assets[name];
    if (!assets.sharedGraph)
        return;

    std::set<std::string> inputs_old, outputs_old, obj_inputs_old, obj_outputs_old;

    std::set<std::string> input_names;
    std::set<std::string> output_names;
    std::set<std::string> obj_input_names;
    std::set<std::string> obj_output_names;
    for (auto param : assets.primitive_inputs) {
        input_names.insert(param.name);
    }
    for (auto param : assets.primitive_outputs) {
        output_names.insert(param.name);
    }

    for (auto param : assets.object_inputs) {
        obj_input_names.insert(param.name);
    }
    for (auto param : assets.object_outputs) {
        obj_output_names.insert(param.name);
    }

    for (const auto& param_name : input_names) {
        inputs_old.insert(param_name);
    }
    for (const auto& param_name : output_names) {
        outputs_old.insert(param_name);
    }

    for (const auto& param_name : obj_input_names) {
        obj_inputs_old.insert(param_name);
    }
    for (const auto& param_name : obj_output_names) {
        obj_outputs_old.insert(param_name);
    }

    params_change_info changes;
    std::map<std::string, bool> paramsIspirm;

    for (auto _pair : info) {
        using T = std::decay_t<decltype(_pair.param)>;
        if (std::holds_alternative<ParamObject>(_pair.param))
        {
            const ParamObject& param = std::get<ParamObject>(_pair.param);
            const std::string oldname = _pair.oldName;
            const std::string newname = param.name;

            auto& in_outputs = param.bInput ? obj_input_names : obj_output_names;
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
                    obj_inputs_old.erase(oldname);
                else
                    obj_outputs_old.erase(oldname);
            }
            else {
                throw makeError<KeyError>(oldname, "the name does not exist on the node");
            }
            paramsIspirm.insert({param.name, false});
        }
        else if (std::holds_alternative<ParamPrimitive>(_pair.param))
        {
            const ParamPrimitive& param = std::get<ParamPrimitive>(_pair.param);
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
            paramsIspirm.insert({ param.name, true });
        }
    }

    //the left names are the names of params which will be removed.
    for (auto rem_name : inputs_old) {
        changes.remove_inputs.insert(rem_name);
    }
    //update the names.
    //input_names.clear();
    //for (const auto& [param, _] : info) {
    //    if (param.bInput)
    //        input_names.insert(param.name);
    //}

    for (auto rem_name : outputs_old) {
        changes.remove_outputs.insert(rem_name);
    }

    for (auto rem_name : obj_inputs_old) {
        changes.remove_inputs.insert(rem_name);
    }

    for (auto rem_name : obj_outputs_old) {
        changes.remove_outputs.insert(rem_name);
    }
    //output_names.clear();
    //for (const auto& [param, _] : info) {
    //    if (!param.bInput)
    //        output_names.insert(param.name);
    //}

    //update subnetnode.
    for (auto name : changes.new_inputs) {
        std::shared_ptr<INode> spNewNode = assets.sharedGraph->createNode("SubInput", name);
        auto it = paramsIspirm.find(name);
        if (it != paramsIspirm.end()) {
            if (it->second) {
                zeno::ParamPrimitive primitive;
                primitive.bInput = false;
                primitive.name = "port";
                primitive.socketType = Socket_Output;
                spNewNode->add_output_prim_param(primitive);
            }
            else {
                zeno::ParamObject paramObj;
                paramObj.bInput = false;
                paramObj.name = "port";
                paramObj.type = Obj_Wildcard;
                paramObj.socketType = zeno::Socket_WildCard;
                spNewNode->add_output_obj_param(paramObj);
            }
            params_change_info changes;
            changes.new_outputs.insert("port");
            changes.outputs.push_back("port");
            changes.outputs.push_back("hasValue");
            spNewNode->update_layout(changes);
        }
    }
    for (const auto& [old_name, new_name] : changes.rename_inputs) {
        assets.sharedGraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_inputs) {
        assets.sharedGraph->removeNode(name);
    }

    for (auto name : changes.new_outputs) {
        std::shared_ptr<INode> spNewNode = assets.sharedGraph->createNode("SubOutput", name);
        auto it = paramsIspirm.find(name);
        if (it != paramsIspirm.end()) {
            if (it->second) {
                zeno::ParamPrimitive primitive;
                primitive.bInput = true;
                primitive.name = "port";
                primitive.type = Param_Wildcard;
                primitive.socketType = Socket_WildCard;
                spNewNode->add_input_prim_param(primitive);
            }
            else {
                zeno::ParamObject paramObj;
                paramObj.bInput = true;
                paramObj.name = "port";
                paramObj.type = Obj_Wildcard;
                paramObj.socketType = zeno::Socket_WildCard;
                spNewNode->add_input_obj_param(paramObj);
            }
            params_change_info changes;
            changes.new_inputs.insert("port");
            changes.inputs.push_back("port");
            spNewNode->update_layout(changes);
        }
    }
    for (const auto& [old_name, new_name] : changes.rename_outputs) {
        assets.sharedGraph->updateNodeName(old_name, new_name);
    }
    for (auto name : changes.remove_outputs) {
        assets.sharedGraph->removeNode(name);
    }

    //update assets data
    assets.primitive_inputs.clear();
    assets.primitive_outputs.clear();    
    assets.object_inputs.clear();
    assets.object_outputs.clear();
    for (auto pair : info) {
        if (auto paramPrim = std::get_if<ParamPrimitive>(&pair.param))
        {
            if (paramPrim->bInput)
                assets.primitive_inputs.push_back(*paramPrim);
            else
                assets.primitive_outputs.push_back(*paramPrim);
        }
        else if (auto paramPrim = std::get_if<ParamObject>(&pair.param))
        {
            if (paramPrim->bInput)
                assets.object_inputs.push_back(*paramPrim);
            else
                assets.object_outputs.push_back(*paramPrim);
        }
    }
    assets.m_customui = customui;
}

std::shared_ptr<Graph> AssetsMgr::forkAssetGraph(std::shared_ptr<Graph> assetGraph, std::shared_ptr<SubnetNode> subNode)
{
    std::shared_ptr<Graph> newGraph = std::make_shared<Graph>(assetGraph->getName(), true);
    newGraph->optParentSubgNode = subNode.get();
    for (const auto& [uuid, spNode] : assetGraph->getNodes())
    {
        zeno::NodeData nodeDat;
        const std::string& name = spNode->get_name();
        const std::string& cls = spNode->get_nodecls();

        if (auto spSubnetNode = std::dynamic_pointer_cast<SubnetNode>(spNode))
        {
            if (m_assets.find(cls) != m_assets.end()) {
                //asset node
                auto spNewSubnetNode = newGraph->createNode(cls, name, true, spNode->get_pos());
            }
            else {
                std::shared_ptr<INode> spNewNode = newGraph->createNode(cls, name);
                nodeDat = spSubnetNode->exportInfo();
                spNewNode->init(nodeDat);   //should clone graph.
            }
        }
        else {
            std::shared_ptr<INode> spNewNode = newGraph->createNode(cls, name);
            nodeDat = spNode->exportInfo();
            spNewNode->init(nodeDat);
        }
    }

    LinksData oldLinks = assetGraph->exportLinks();
    for (zeno::EdgeInfo oldLink : oldLinks) {
        newGraph->addLink(oldLink);
    }
    return newGraph;
}

void AssetsMgr::initAssetSubInputOutput(Asset& newAsst)
{
    std::shared_ptr<zeno::INode> input1Node = newAsst.sharedGraph->getNode("input1");
    zeno::ParamPrimitive paramInput;
    paramInput.bInput = false;
    paramInput.name = "port";
    zeno::PrimVar def = int(0);
    paramInput.defl = zeno::reflect::make_any<zeno::PrimVar>(def);
    paramInput.type = zeno::types::gParamType_Int;
    paramInput.bSocketVisible = false;
    input1Node->add_output_prim_param(paramInput);
    std::shared_ptr<zeno::INode> output1Node = newAsst.sharedGraph->getNode("output1");
    zeno::ParamPrimitive paramOutput;
    paramOutput.bInput = true;
    paramOutput.name = "port";
    paramOutput.type = Param_Wildcard;
    paramOutput.socketType = Socket_WildCard;
    output1Node->add_input_prim_param(paramOutput);
    std::shared_ptr<zeno::INode> objInput1Node = newAsst.sharedGraph->getNode("objInput1");
    zeno::ParamObject paramObj;
    paramObj.bInput = false;
    paramObj.name = "port";
    paramObj.type = Obj_Wildcard;
    paramObj.socketType = zeno::Socket_WildCard;
    objInput1Node->add_output_obj_param(paramObj);
    std::shared_ptr<zeno::INode> objOutput1Node = newAsst.sharedGraph->getNode("objOutput1");
    paramObj.bInput = true;
    objOutput1Node->add_input_obj_param(paramObj);
}

ZENO_API bool AssetsMgr::isAssetGraph(std::shared_ptr<Graph> spGraph) const
{
    for (auto& [name, asset] : m_assets) {
        if (asset.sharedGraph == spGraph)
            return true;
    }
    return false;
}

ZENO_API bool AssetsMgr::generateAssetName(std::string& name)
{
    std::string new_name = name;
    if (m_assets.find(new_name) == m_assets.end()) {
        return false;
    }
    int i = 1;
    while (m_assets.find(new_name) != m_assets.end())
    {
        new_name = name + "(" + std::to_string(i++) + ")";
    }
    name = new_name;
    return true;
}

ZENO_API std::shared_ptr<INode> AssetsMgr::newInstance(std::shared_ptr<Graph> pGraph, const std::string& assetName, const std::string& nodeName, bool createInAsset) {
    if (m_assets.find(assetName) == m_assets.end()) {
        return nullptr;
    }

    Asset& assets = m_assets[assetName];
    if (!assets.sharedGraph) {
        getAssetGraph(assetName, true);
    }
    assert(assets.sharedGraph);

    std::shared_ptr<SubnetNode> spNode = std::make_shared<SubnetNode>();
    spNode->initUuid(pGraph, assetName);
    std::shared_ptr<Graph> assetGraph;
    if (!createInAsset) {
        //should expand the asset graph into a tree.
        assetGraph = forkAssetGraph(assets.sharedGraph, spNode);
    }
    else {
        assetGraph = assets.sharedGraph;
    }

    spNode->subgraph = assetGraph;
    spNode->set_name(nodeName);
    spNode->m_customUi = assets.m_customui;

    for (const ParamPrimitive& param : assets.primitive_inputs)
    {
        spNode->add_input_prim_param(param);
    }

    for (const ParamPrimitive& param : assets.primitive_outputs)
    {
        spNode->add_output_prim_param(param);
    }

    for (const auto& param : assets.object_inputs)
    {
        spNode->add_input_obj_param(param);
    }

    for (const auto& param : assets.object_outputs)
    {
        spNode->add_output_obj_param(param);
    }

    return std::dynamic_pointer_cast<INode>(spNode);
}

}