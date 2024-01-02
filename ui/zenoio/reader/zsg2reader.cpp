#include "zsg2reader.h"
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "iotags.h"
#include <fstream>
#include <filesystem>
#include <zenoio/include/iohelper.h>
#include <zeno/utils/helper.h>


using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace zenoio {

Zsg2Reader::Zsg2Reader() : m_bDiskReading(true), m_ioVer(zeno::VER_3) {}

Zsg2Reader& Zsg2Reader::getInstance()
{
    static Zsg2Reader reader;
    return reader;
}

bool Zsg2Reader::openFile(const std::string& fn, zeno::ZSG_PARSE_RESULT& result)
{
    std::filesystem::path filePath(fn);
    if (!std::filesystem::exists(filePath)) {
        zeno::log_error("cannot open zsg file: {} ({})", fn);
        return false;
    }

    rapidjson::Document doc;

    auto szBuffer = std::filesystem::file_size(filePath);
    if (szBuffer == 0)
    {
        zeno::log_error("the zsg file is a empty file");
        return false;
    }

    std::vector<char> dat(szBuffer);
    FILE* fp = fopen(filePath.string().c_str(), "rb");
    if (!fp) {
        zeno::log_error("zsg file does not exist");
        return false;
    }
    size_t ret = fread(&dat[0], 1, szBuffer, fp);
    assert(ret == szBuffer);
    fclose(fp);
    fp = nullptr;

    doc.Parse(&dat[0]);

    m_ioVer = zeno::VER_3;
    if (doc.HasMember("version") && doc["version"].IsString())
    {
        std::string ver = doc["version"].GetString();
        if (ver == "v2")
            m_ioVer = zeno::VER_2;
        else if (ver == "v2.5")
            m_ioVer = zeno::VER_2_5;
        else if (ver == "v3.0")
            m_ioVer = zeno::VER_3;
    }
    else {
        zeno::log_warn("unknown io foramt for current zsg");
    }

    if (!doc.IsObject())
    {
        zeno::log_error("zsg json file is corrupted");
        return false;
    }

    if ((m_ioVer != zeno::VER_3 && !doc.HasMember("graph")) ||
        (m_ioVer == zeno::VER_3 && (!doc.HasMember("main") || !doc.HasMember("subgraphs"))))
    {
        return false;
    }

    const rapidjson::Value &graph = doc.HasMember("graph") ? doc["graph"] : doc["subgraphs"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}", fn);
        return false;
    }

    if (!doc.HasMember("descs"))
    {
        zeno::log_error("there is not descs in current zsg");
        return false;
    }

    zeno::NodeDescs nodesDescs = _parseDescs(doc["descs"]);
    if (!ret) {
        return false;
    }

    zeno::AssetsData subgraphDatas;
    //init keys
    for (const auto& subgraph : graph.GetObject())
    {
        const std::string &graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;

        zeno::ZenoAsset asset;
        if (nodesDescs.find(graphName) != nodesDescs.end())
            asset.desc = nodesDescs[graphName];

        asset.graph = zeno::GraphData();
        subgraphDatas[graphName] = asset;
    }

    //zsg3.0以下的格式，子图将加入并成为项目的资产
    for (const auto& subgraph : graph.GetObject())
    {
        const std::string& graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;
        if (!_parseSubGraph(graphName,
                    subgraph.value,
                    subgraphDatas,
                    subgraphDatas[graphName].graph))
        {
            return false;
        }
    }

    zeno::GraphData mainData;
    if (doc.HasMember("main") || graph.HasMember("main"))
    {
        const rapidjson::Value& mainGraph = doc.HasMember("main") ? doc["main"] : graph["main"];
        if (!_parseSubGraph("/main", mainGraph, subgraphDatas, mainData))
            return false;
    }

    if (doc.HasMember("views"))
    {
        _parseViews(doc["views"], result);
    }

    result.iover = m_ioVer;
    result.descs = nodesDescs;
    result.mainGraph = mainData;
    result.assetGraphs = subgraphDatas;
    return true;
}

bool Zsg2Reader::_parseSubGraph(
            const std::string& graphPath,
            const rapidjson::Value& subgraph,
            const zeno::AssetsData& subgraphDatas,
            zeno::GraphData& subgData)
{
    if (!subgraph.IsObject() || !subgraph.HasMember("nodes"))
        return false;

    const auto& nodes = subgraph["nodes"];
    if (nodes.IsNull())
        return false;

    for (const auto& node : nodes.GetObject())
    {
        const std::string& nodeid = node.name.GetString();
        const zeno::NodeData& nodeData = _parseNode(graphPath, nodeid, node.value, subgraphDatas, subgData.links);
        subgData.nodes.insert(std::make_pair(nodeid, nodeData));
    }
    return true;
}

zeno::NodeData Zsg2Reader::_parseNode(
                    const std::string& subgPath,
                    const std::string& nodeid,
                    const rapidjson::Value& nodeObj,
                    const zeno::AssetsData& subgraphDatas,
                    zeno::LinksData& links)
{
    zeno::NodeData retNode;

    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const std::string& name = nameValue.GetString();

    retNode.ident = nodeid;
    retNode.cls = name;

    std::string customName;
    if (objValue.HasMember("customName")) {
        const std::string &tmp = objValue["customName"].GetString();
        customName = tmp;
    }
    retNode.name = customName;

    bool isParsingAssets = subgPath.rfind("/main", 0) != 0;

    //should expand the subgraph node recursively.
    if (subgraphDatas.find(name) != subgraphDatas.end())
    {
        if (!isParsingAssets)
        {
            retNode.subgraph = zenoio::fork(subgPath + "/" + nodeid, subgraphDatas, name);
        }
    }

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(subgPath, nodeid, name, objValue["inputs"], retNode, links);
    }
    if (objValue.HasMember("params"))
    {
        _parseParams(nodeid, name, objValue["params"], retNode);
    }
    if (objValue.HasMember("outputs"))
    {
        _parseOutputs(nodeid, name, objValue["outputs"], retNode, links);
    }
    if (objValue.HasMember("customui-panel"))
    {
        _parseCustomPanel(nodeid, name, objValue["customui-panel"], retNode);
    }

    if (objValue.HasMember("uipos"))
    {
        auto uipos = objValue["uipos"].GetArray();
        retNode.uipos = { uipos[0].GetFloat(), uipos[1].GetFloat() };
    }
    if (objValue.HasMember("options"))
    {
        auto optionsArr = objValue["options"].GetArray();
        zeno::NodeStatus opts = zeno::None;
        for (int i = 0; i < optionsArr.Size(); i++)
        {
            assert(optionsArr[i].IsString());

            const std::string& optName = optionsArr[i].GetString();
            if (optName == "ONCE") {} //deprecated 
            else if (optName == "PREP") {} //deprecated 
            else if (optName == "VIEW") { opts |= zeno::View; }
            else if (optName == "MUTE") { opts |= zeno::Mute; }
            else if (optName == "CACHE") {} //deprecated 
            else if (optName == "collapsed") {}
        }
        retNode.status = opts;
    }
    if (objValue.HasMember("children"))
    {
        //这个其实是zsg3.0的属性了，而且zsg3.0不会把资产的节点写到io，除非用户fork并修改了
        /*
        zeno::GraphData subg;
        _parseSubGraph(subgPath + '/' + nodeid, objValue["children"], legacyDescs, subgraphDatas, subg);
        ret.children = subg.nodes;
        links.append(subg.links);
        */
    }

    if (name == "Blackboard")
    {
        zeno::GroupInfo blackboard;
        //use subkey "blackboard" for zeno2 io, but still compatible with zeno1
        const rapidjson::Value &blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

        if (blackBoardValue.HasMember("special")) {
            blackboard.special = blackBoardValue["special"].GetBool();
        }

        blackboard.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
        blackboard.content = blackBoardValue.HasMember("content") ? blackBoardValue["content"].GetString() : "";

        if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
            float w = blackBoardValue["width"].GetFloat();
            float h = blackBoardValue["height"].GetFloat();
            blackboard.sz = { w,h };
        }
        if (blackBoardValue.HasMember("params")) {
            //todo
        }
        //TODO: import blackboard.
    }
    else if (name == "Group") 
    {
        zeno::GroupInfo group;
        const rapidjson::Value &blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

        group.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
        group.background = blackBoardValue.HasMember("background") ? blackBoardValue["background"].GetString() : "#3C4645";

        if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
            float w = blackBoardValue["width"].GetFloat();
            float h = blackBoardValue["height"].GetFloat();
            group.sz = { w,h };
        }
        if (blackBoardValue.HasMember("items")) {
            auto item_keys = blackBoardValue["items"].GetArray();
            for (int i = 0; i < item_keys.Size(); i++) {
                std::string key = item_keys[i].GetString();
                group.items.push_back(key);
            }
        }
        //TODO: import group.
    }

    return retNode;
}

void Zsg2Reader::_parseChildNodes(
                    const std::string& rootPath,
                    const rapidjson::Value& jsonNodes,
                    const zeno::NodeDescs& descriptors,
                    zeno::NodeData& ret)
{
    if (!jsonNodes.HasMember("nodes"))
        return;
}

void Zsg2Reader::_parseViews(const rapidjson::Value& jsonViews, zeno::ZSG_PARSE_RESULT& res)
{
    if (jsonViews.HasMember("timeline"))
    {
        _parseTimeline(jsonViews["timeline"], res);
    }
}

void Zsg2Reader::_parseTimeline(const rapidjson::Value& jsonTimeline, zeno::ZSG_PARSE_RESULT& res)
{
    assert(jsonTimeline.HasMember(timeline::start_frame) && jsonTimeline[timeline::start_frame].IsInt());
    assert(jsonTimeline.HasMember(timeline::end_frame) && jsonTimeline[timeline::end_frame].IsInt());
    assert(jsonTimeline.HasMember(timeline::curr_frame) && jsonTimeline[timeline::curr_frame].IsInt());
    assert(jsonTimeline.HasMember(timeline::always) && jsonTimeline[timeline::always].IsBool());

    res.timeline.beginFrame = jsonTimeline[timeline::start_frame].GetInt();
    res.timeline.endFrame = jsonTimeline[timeline::end_frame].GetInt();
    res.timeline.currFrame = jsonTimeline[timeline::curr_frame].GetInt();
    res.timeline.bAlways = jsonTimeline[timeline::always].GetBool();
}

void Zsg2Reader::_parseInputs(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeName,
                const rapidjson::Value& inputs,
                zeno::NodeData& ret,
                zeno::LinksData& links)
{
    for (const auto& inObj : inputs.GetObject())
    {
        const std::string& inSock = inObj.name.GetString();
        const auto& inputObj = inObj.value;
        if (inputObj.IsNull())
        {
            zeno::ParamInfo param;
            param.name = inSock;
            ret.inputs.push_back(param);
        }
        else if (inputObj.IsObject())
        {
            zeno::ParamInfo param = _parseSocket(subgPath, id, nodeName, inSock, true, inputObj, links);
            ret.inputs.push_back(param);
        }
        else
        {
            zeno::log_error("unknown format");
        }
    }
}

zeno::ParamInfo Zsg2Reader::_parseSocket(
        const std::string& subgPath,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& sockName,
        bool bInput,
        const rapidjson::Value& sockObj,
        zeno::LinksData& links)
{
    zeno::ParamInfo param;

    std::string sockProp;
    if (sockObj.HasMember("property"))
    {
        //ZASSERT_EXIT(sockObj["property"].IsString());
        sockProp = sockObj["property"].GetString();
    }

    zeno::ParamControl ctrl = zeno::NullControl;

    zeno::SocketProperty prop = zeno::SocketProperty::Socket_Normal;
    if (sockProp == "dict-panel")
        prop = zeno::SocketProperty::Socket_Normal;    //deprecated
    else if (sockProp == "editable")
        prop = zeno::SocketProperty::Socket_Editable;
    else if (sockProp == "group-line")
        prop = zeno::SocketProperty::Socket_Normal;    //deprecated

    param.prop = prop;
    param.name = sockName;

    if (m_bDiskReading &&
        (prop == zeno::SocketProperty::Socket_Editable ||
         nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
    {
        if (prop == zeno::SocketProperty::Socket_Editable) {
            //like extract dict.
            param.type = zeno::Param_String;
        } else {
            param.type = zeno::Param_Null;
        }
    }

    if (sockObj.HasMember("type") && sockObj.HasMember("default-value")) {
        param.type = zeno::convertToType(sockObj["type"].GetString());
        param.defl = zenoio::jsonValueToZVar(sockObj["default-value"], param.type);
    }

    //link:
    if (bInput && sockObj.HasMember("link") && sockObj["link"].IsString())
    {
        std::string outLinkPath = sockObj["link"].GetString();

        auto lst = zeno::split_str(outLinkPath, ':');
        if (lst.size() > 2)
        {
            const std::string& outId = lst[1];
            const std::string& fuckingpath = lst[2];
            lst = zeno::split_str(fuckingpath, '/');
            if (lst.size() > 2) {
                std::string group = lst[1];
                std::string param = lst[2];
                std::string key;
                if (lst.size() > 3)
                    key = lst[3];
                zeno::EdgeInfo edge = { outId, param, key, id, sockName, "" };
                links.push_back(edge);
            }
        }
    }

    if (sockObj.HasMember("dictlist-panel"))
    {
        _parseDictPanel(subgPath, bInput, sockObj["dictlist-panel"], id, sockName, nodeCls, links);
    }

    if (sockObj.HasMember("control"))
	{
        zeno::ParamControl ctrl;
        zeno::ControlProperty props;
        bool bret = zenoio::importControl(sockObj["control"], ctrl, props);
        if (bret) {
            param.control = ctrl;
            if (props.items || props.ranges)
                param.ctrlProps = props;
        }
    }

    if (sockObj.HasMember("tooltip")) 
    {
        param.tooltip = sockObj["tooltip"].GetString();
    }
}

zeno::NodesData Zsg2Reader::_parseChildren(const rapidjson::Value& jsonNodes)
{
    //there is no children concept in zsg2.5
    zeno::NodesData children;
    //_parseSubGraph(, , , children);
    return children;
}

void Zsg2Reader::_parseDictPanel(
            const std::string& subgPath,
            bool bInput,
            const rapidjson::Value& dictPanelObj, 
            const std::string& id,
            const std::string& sockName,
            const std::string& nodeName,
            zeno::LinksData& links)
{
    if (dictPanelObj.HasMember("collasped") && dictPanelObj["collasped"].IsBool())
    {
        //deprecated.
    }
    if (dictPanelObj.HasMember("keys"))
    {
        const rapidjson::Value& dictKeys = dictPanelObj["keys"];
        for (const auto& kv : dictKeys.GetObject())
        {
            const std::string& keyName = kv.name.GetString();
            const rapidjson::Value& inputObj = kv.value;

            std::string link;
            if (inputObj.HasMember("link") && inputObj["link"].IsString())
            {
                std::string link = inputObj["link"].GetString();
                //EdgeInfo edge;

                auto lst = zeno::split_str(link, ':');
                if (lst.size() > 2)
                {
                    const std::string& outId = lst[1];
                    const std::string& fuckingpath = lst[2];
                    lst = zeno::split_str(fuckingpath, '/');
                    if (lst.size() > 2) {
                        std::string group = lst[1];
                        std::string param = lst[2];
                        std::string key;
                        if (lst.size() > 3)
                            key = lst[3];
                        zeno::EdgeInfo edge = { outId, param, key, id, sockName, keyName };
                        links.push_back(edge);
                    }
                }
            }

            //ignore output case because there is no link on output socket.
        }
    }
}

void Zsg2Reader::_parseOutputs(
        const std::string &id,
        const std::string &nodeName,
        const rapidjson::Value& outputs,
        zeno::NodeData& ret,
        zeno::LinksData& links)
{
    for (const auto& outParamObj : outputs.GetObject())
    {
        const std::string& outParam = outParamObj.name.GetString();
        const auto& outObj = outParamObj.value;
        if (outObj.IsNull())
        {
            zeno::ParamInfo param;
            param.name = outParam;
            ret.outputs.push_back(param);
        }
        else if (outObj.IsObject())
        {
            zeno::ParamInfo param = _parseSocket("", id, nodeName, outParam, true, outObj, links);
            ret.outputs.push_back(param);
        }
        else
        {
            zeno::log_error("unknown format");
        }
    }
}

bool Zsg2Reader::_parseParams(const std::string& id, const std::string& nodeCls, const rapidjson::Value& jsonParams, zeno::NodeData& ret)
{
    if (jsonParams.IsObject()) {
        //PARAMS_INFO params;
        for (const auto& paramObj : jsonParams.GetObject()) {
            const std::string& name = paramObj.name.GetString();
            const rapidjson::Value& valueObj = paramObj.value;
            if (!valueObj.IsObject() || !valueObj.HasMember(iotags::params::params_valueKey)) //compatible old version
                return false;

            zeno::ParamInfo param;

            param.name = name;

            if (valueObj.HasMember("type"))
            {
                param.type = zeno::convertToType(valueObj["type"].GetString());
            }

            //它不知道会不会和SubInput的type参数冲突，这个很讨厌，这里直接解析算了，放弃历史包袱
            param.defl = zenoio::jsonValueToZVar(valueObj[iotags::params::params_valueKey], param.type);

            if (valueObj.HasMember("control"))
            {
                zeno::ParamControl ctrl;
                zeno::ControlProperty props;
                bool bret = zenoio::importControl(valueObj["control"], ctrl, props);
                if (bret) {
                    param.control = ctrl;
                    if (props.items || props.ranges)
                        param.ctrlProps = props;
                }
            }

            if (valueObj.HasMember("tooltip"))
            {
                std::string toolTip(valueObj["tooltip"].GetString());
                param.tooltip = toolTip;
            }
            ret.inputs.push_back(param);
        }
    }
    return true;
}


void Zsg2Reader::_parseCustomPanel(const std::string& id, const std::string& nodeName, const rapidjson::Value& jsonCutomUI, zeno::NodeData& ret)
{
    //VPARAM_INFO invisibleRoot = zenomodel::importCustomUI(jsonCutomUI);
    //ret.customPanel = invisibleRoot;
    //TODO
}

zeno::NodeDescs Zsg2Reader::_parseDescs(const rapidjson::Value& jsonDescs)
{
    zeno::NodeDescs _descs;     //不需要系统内置节点的desc，只要读文件的就可以
    zeno::LinksData lnks;       //没用的
    for (const auto& node : jsonDescs.GetObject())
    {
        const std::string& nodeCls = node.name.GetString();
        const auto& objValue = node.value;

        zeno::NodeDesc desc;
        desc.name = nodeCls;
        if (objValue.HasMember("inputs"))
        {
            if (objValue["inputs"].IsArray()) 
            {
                //系统节点导出的描述，形如：
                /*
                "inputs": [
                    [
                        "ListObject",
                        "keys",
                        ""
                    ],
                    [
                        "",
                        "SRC",
                        ""
                    ]
                ],
                */
                auto inputs = objValue["inputs"].GetArray();
                for (int i = 0; i < inputs.Size(); i++) 
                {
                    if (inputs[i].IsArray())
                    {
                        auto input_triple = inputs[i].GetArray();
                        std::string socketType, socketName, socketDefl;
                        if (input_triple.Size() > 0 && input_triple[0].IsString())
                            socketType = input_triple[0].GetString();
                        if (input_triple.Size() > 1 && input_triple[1].IsString())
                            socketName = input_triple[1].GetString();
                        if (input_triple.Size() > 2 && input_triple[2].IsString())
                            socketDefl = input_triple[2].GetString();

                        if (!socketName.empty())
                        {
                            zeno::ParamInfo param;
                            param.name = socketName;
                            param.type = zeno::convertToType(socketDefl);
                            param.defl = socketDefl;    //不转了，太麻烦了。..反正普通节点的desc也只是参考

                            desc.inputs.push_back(param);
                        }
                    }
                }
            } 
            else if (objValue["inputs"].IsObject()) 
            {
                auto inputs = objValue["inputs"].GetObject();
                for (const auto &input : inputs)
                {
                    std::string socketName = input.name.GetString();
                    _parseSocket("", "", nodeCls, socketName, true, input.value, lnks);
                }
            }
        }
        if (objValue.HasMember("params"))
        {
            if (objValue["params"].IsArray()) 
            {
                auto params = objValue["params"].GetArray();
                for (int i = 0; i < params.Size(); i++) 
                {
                    if (params[i].IsArray()) {
                        auto param_triple = params[i].GetArray();
                        std::string socketType, socketName, socketDefl;

                        if (param_triple.Size() > 0 && param_triple[0].IsString())
                            socketType = param_triple[0].GetString();
                        if (param_triple.Size() > 1 && param_triple[1].IsString())
                            socketName = param_triple[1].GetString();
                        if (param_triple.Size() > 2 && param_triple[2].IsString())
                            socketDefl = param_triple[2].GetString();

                        if (!socketName.empty())
                        {
                            zeno::ParamInfo param;
                            param.name = socketName;
                            param.type = zeno::convertToType(socketDefl);
                            param.defl = socketDefl;    //不转了，太麻烦了。..反正普通节点的desc也只是参考
                            desc.inputs.push_back(param);
                        }
                    }
                }
            } 
            else if (objValue["params"].IsObject()) 
            {
                auto params = objValue["params"].GetObject();
                for (const auto &param : params) 
                {
                    std::string socketName = param.name.GetString();
                    _parseSocket("", "", nodeCls, socketName, true, param.value, lnks);
                }
            }
        }
        if (objValue.HasMember("outputs"))
        {
            if (objValue["outputs"].IsArray()) 
            {
                auto outputs = objValue["outputs"].GetArray();
                for (int i = 0; i < outputs.Size(); i++)
                {
                    if (outputs[i].IsArray()) {
                        auto output_triple = outputs[i].GetArray();
                        std::string socketType, socketName, socketDefl;

                        if (output_triple.Size() > 0 && output_triple[0].IsString())
                            socketType = output_triple[0].GetString();
                        if (output_triple.Size() > 1 && output_triple[1].IsString())
                            socketName = output_triple[1].GetString();
                        if (output_triple.Size() > 2 && output_triple[2].IsString())
                            socketDefl = output_triple[2].GetString();

                        if (!socketName.empty())
                        {
                            zeno::ParamInfo param;
                            param.name = socketName;
                            param.type = zeno::convertToType(socketDefl);
                            param.defl = socketDefl;
                            desc.outputs.push_back(param);
                        }
                    }
                }
            } 
            else if (objValue["outputs"].IsObject()) 
            {
                auto outputs = objValue["outputs"].GetObject();
                for (const auto &output : outputs) 
                {
                    std::string socketName = output.name.GetString();
                    _parseSocket("", "", nodeCls, socketName, false, output.value, lnks);
                }
            }
        }
        if (objValue.HasMember("categories") && objValue["categories"].IsArray())
        {
            auto categories = objValue["categories"].GetArray();
            for (int i = 0; i < categories.Size(); i++)
            {
                desc.categories.push_back(categories[i].GetString());
            }
        }

        _descs.insert(std::make_pair(nodeCls, desc));
    }
    return _descs;
}



}
