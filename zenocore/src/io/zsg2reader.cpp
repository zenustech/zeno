#include <zeno/io/zsg2reader.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/io/iotags.h>
#include <fstream>
#include <filesystem>
#include <zeno/io/iohelper.h>
#include <zeno/utils/helper.h>
#include "reflect/reflection.generated.hpp"


namespace zenoio {

ZENO_API Zsg2Reader::Zsg2Reader() {}

bool Zsg2Reader::_parseMainGraph(const rapidjson::Document& doc, zeno::GraphData& mainData)
{
    if (doc.HasMember("version") && doc["version"].IsString())
    {
        std::string ver = doc["version"].GetString();
        if (ver == "v2")
            m_ioVer = zeno::VER_2;
        else if (ver == "v2.5")
            m_ioVer = zeno::VER_2_5;
    }
    else {
        zeno::log_warn("unknown io foramt for current zsg");
    }

    if (m_ioVer != zeno::VER_3 && !doc.HasMember("graph"))
    {
        return false;
    }

    const rapidjson::Value& graph = doc.HasMember("graph") ? doc["graph"] : doc["subgraphs"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}");
        return false;
    }

    if (!doc.HasMember("descs"))
    {
        zeno::log_error("there is not descs in current zsg");
        return false;
    }

    //seems useless to parse desc.
    //zeno::NodeDescs nodesDescs = _parseDescs(doc["descs"]);

    //zeno::AssetsData subgraphDatas;
    std::map<std::string, zeno::GraphData> sharedSubg;

    //init keys
    for (const auto& subgraph : graph.GetObject())
    {
        const std::string& graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;

        sharedSubg[graphName] = zeno::GraphData();
        sharedSubg[graphName].templateName = graphName;
    }

    //zsg3.0以下的格式，子图直接成为Subnet递归展开
    for (const auto& subgraph : graph.GetObject())
    {
        const std::string& graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;
        if (!_parseSubGraph(graphName,
            subgraph.value,
            sharedSubg,
            sharedSubg[graphName]))
        {
            return false;
        }
    }

    mainData.name = mainData.templateName = "main";
    if (doc.HasMember("main") || graph.HasMember("main"))
    {
        const rapidjson::Value& mainGraph = doc.HasMember("main") ? doc["main"] : graph["main"];
        if (!_parseSubGraph("/main", mainGraph, sharedSubg, mainData))
            return false;
    }

    return true;
}

bool Zsg2Reader::_parseSubGraph(
            const std::string& graphPath,
            const rapidjson::Value& subgraph,
            const std::map<std::string, zeno::GraphData>& subgraphDatas,
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
        subgData.nodes.insert(std::make_pair(nodeData.name, nodeData));
    }
    return true;
}

zeno::NodeData Zsg2Reader::_parseNode(
                    const std::string& subgPath,
                    const std::string& nodeid,
                    const rapidjson::Value& nodeObj,
                    const std::map<std::string, zeno::GraphData>& sharedSubg,
                    zeno::LinksData& links)
{
    zeno::NodeData retNode;

    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const std::string& cls = nameValue.GetString();

    retNode.name = nodeid;
    retNode.cls = cls;

    bool isParsingSubg = subgPath.rfind("/main", 0) != 0;

    //should expand the subgraph node recursively.
    if (sharedSubg.find(cls) != sharedSubg.end())
    {
        retNode.cls = "Subnet";
        if (isParsingSubg)
        {
            retNode.subgraph = zeno::GraphData();
            retNode.subgraph->templateName = cls;
        }
        else
        {
            retNode.subgraph = zenoio::fork(sharedSubg, cls);
            retNode.subgraph->name = retNode.name;
        }
    }

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(nodeid, cls, objValue["inputs"], retNode, links);
    }
    if (objValue.HasMember("params"))
    {
        _parseParams(nodeid, cls, objValue["params"], retNode);
    }
    if (objValue.HasMember("outputs"))
    {
        _parseOutputs(nodeid, cls, objValue["outputs"], retNode, links);
    }
    if (objValue.HasMember("customui-panel"))
    {
        //deprecated legacy customui.
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
            else if (optName == "VIEW") { retNode.bView = true; }
            else if (optName == "MUTE") { }
            else if (optName == "CACHE") {} //deprecated 
            else if (optName == "collapsed") {}
        }
    }

    if (cls == "Blackboard")
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
        retNode.group = blackboard;
        //TODO: import blackboard.
    }
    else if (cls == "Group")
    {
        zeno::GroupInfo group;
        const rapidjson::Value &blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

        group.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
        std::string color = blackBoardValue.HasMember("background") ? blackBoardValue["background"].GetString() : "#3C4645";
        color.replace(0, 1, "");
        int num = std::stoi(color, NULL, 16);
        int red = num >> 16 & 0xFF;
        int green = num >> 8 & 0xFF;
        int blue = num & 0xFF;
        group.background[0] = red / 255.0;
        group.background[1] = green / 255.0;
        group.background[2] = blue / 255.0;

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
        retNode.group = group;
        //TODO: import group.
    }

    return retNode;
}

void Zsg2Reader::_parseSocket(
        const bool bInput,
        const bool bSubnetNode,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& sockName,
        const rapidjson::Value& sockObj,
        zeno::NodeData& ret,
        zeno::LinksData& links)
{
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

    zeno::SocketType socketType = zeno::NoSocket;
    zeno::ParamType paramType = Param_Null;
    zeno::reflect::Any defl;
    std::string tooltip;

    if (m_bDiskReading &&
        (prop == zeno::SocketProperty::Socket_Editable ||
         nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
    {
        if (prop == zeno::SocketProperty::Socket_Editable) {
            //like extract dict.
            paramType = Param_String;
        } else {
            paramType = Param_Null;
        }
    }

    if (sockObj.HasMember("type") && sockObj.HasMember("default-value")) {
        paramType = zeno::convertToType(sockObj["type"].GetString());
        defl = zenoio::jsonValueToAny(sockObj["default-value"], paramType);
    }
    if (!bInput && paramType == Param_Null)
    {
        auto& nodeClass = zeno::getSession().nodeClasses;
        auto it = nodeClass.find(nodeCls);
        if (it != nodeClass.end()) {
            const auto& outputs = it->second->m_customui.outputPrims;
            for (const auto& output : outputs)
            {
                if (output.name == sockName)
                {
                    paramType = output.type;
                    break;
                }
            }
        }
    }
    bool bPrimType = zeno::isPrimitiveType(paramType);
    if (bPrimType) {
        socketType = zeno::Socket_Primitve;
    }
    else {
        if (bInput) {
            //这种情况大概率是连对象，默认赋予Owing端口吧
            socketType = zeno::Socket_Owning;
        }
        else {
            socketType = zeno::Socket_Output;
        }
    }

    //link:
    if (bInput && sockObj.HasMember("link") && sockObj["link"].IsString())
    {
        std::string outLinkPath = sockObj["link"].GetString();

        auto lst = zeno::split_str(outLinkPath, ':');
        if (lst.size() > 2)
        {
            const std::string outId = lst[1];
            const std::string fuckingpath = lst[2];
            lst = zeno::split_str(fuckingpath, '/');
            if (lst.size() > 2) {
                std::string group = lst[1];
                std::string param = lst[2];
                std::string key;
                if (lst.size() > 3)
                    key = lst[3];
                if (param != "DST") {
                    zeno::EdgeInfo edge = { outId, param, key, id, sockName, "" };
                    links.push_back(edge);
                }
            }
        }
    }

    if (sockObj.HasMember("dictlist-panel"))
    {
        _parseDictPanel(bInput, sockObj["dictlist-panel"], id, sockName, nodeCls, links);
    }

    zeno::reflect::Any ctrlProps;

    if (sockObj.HasMember("control"))
    {
        zenoio::importControl(sockObj["control"], ctrl, ctrlProps);
    }

    if (sockObj.HasMember("tooltip")) 
    {
        tooltip = sockObj["tooltip"].GetString();
    }

    if (bPrimType) {
        zeno::ParamPrimitive param;
        param.bInput = bInput;
        param.control = ctrl;
        param.ctrlProps = ctrlProps;
        param.defl = defl;
        param.name = sockName;
        param.prop = prop;
        param.socketType = socketType;
        param.tooltip = tooltip;
        param.type = paramType;
        if (bInput) {
            //老zsg没有层级结构，直接用默认就行
            if (ret.customUi.inputPrims.tabs.empty())
            {
                zeno::ParamTab tab;
                tab.name = "Tab1";
                zeno::ParamGroup group;
                group.name = "Group1";
                tab.groups.emplace_back(group);
                ret.customUi.inputPrims.tabs.emplace_back(tab);
            }
            auto& group = ret.customUi.inputPrims.tabs[0].groups[0];
            group.params.emplace_back(param);
        }
        else {
            ret.customUi.outputPrims.emplace_back(param);
        }
    }
    else {
        zeno::ParamObject param;
        param.bInput = bInput;
        param.name = sockName;
        param.prop = prop;
        param.socketType = socketType;
        param.tooltip = tooltip;
        param.type = paramType;
        if (bInput) {
            ret.customUi.inputObjs.emplace_back(param);
        }
        else {
            ret.customUi.outputObjs.emplace_back(param);
        }
    }
}

void Zsg2Reader::_parseDictPanel(
            bool bInput,
            const rapidjson::Value& dictPanelObj, 
            const std::string& id,
            const std::string& sockName,
            const std::string& nodeName,
            zeno::LinksData& links)
{
    if (!bInput)
    {
        //no need to parse output dict keys, because we can get it from input links from these output dict sockets.
        return;
    }
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
                    const std::string outId = lst[1];
                    const std::string fuckingpath = lst[2];
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

            zeno::ParamPrimitive param;

            param.name = name;

            if (valueObj.HasMember("type"))
            {
                param.type = zeno::convertToType(valueObj["type"].GetString());
            }

            //它不知道会不会和SubInput的type参数冲突，这个很讨厌，这里直接解析算了，放弃历史包袱
            param.defl = zenoio::jsonValueToAny(valueObj[iotags::params::params_valueKey], param.type);
            param.socketType = zeno::NoSocket; //以前的定义是不包含这个的。

            if (valueObj.HasMember("control"))
            {
                zeno::ParamControl ctrl;
                zenoio::importControl(valueObj["control"], ctrl, param.ctrlProps);
            }

            if (valueObj.HasMember("tooltip"))
            {
                std::string toolTip(valueObj["tooltip"].GetString());
                param.tooltip = toolTip;
            }
            //how to place?
            //ret.inputs.push_back(param);
        }
    }
    return true;
}

}
