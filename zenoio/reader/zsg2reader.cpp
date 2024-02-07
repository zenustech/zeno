#include "zsg2reader.h"
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "include/iotags.h"
#include <fstream>
#include <filesystem>
#include <zenoio/include/iohelper.h>
#include <zeno/utils/helper.h>


using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace zenoio {

Zsg2Reader::Zsg2Reader() {}

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

    //zsg3.0以下的格式，子图将加入并成为项目的资产
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
        subgData.nodes.insert(std::make_pair(nodeid, nodeData));
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
            else if (optName == "VIEW") { opts = opts | zeno::View; }
            else if (optName == "MUTE") { opts = opts | zeno::Mute; }
            else if (optName == "CACHE") {} //deprecated 
            else if (optName == "collapsed") {}
        }
        retNode.status = opts;
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
        //TODO: import blackboard.
    }
    else if (cls == "Group")
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

zeno::ParamInfo Zsg2Reader::_parseSocket(
        const bool bInput,
        const std::string& id,
        const std::string& nodeCls,
        const std::string& sockName,
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
    return param;
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

}
