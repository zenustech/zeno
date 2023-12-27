#include "zsg2reader.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/customui/customuirw.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "zenoedit/util/log.h"
#include "variantptr.h"
#include "common.h"
#include <zenomodel/customui/customuirw.h>
#include <zenomodel/include/nodesmgr.h>
#include "iotags.h"
#include <zenomodel/include/graphsmanagment.h>
#include <common_def.h>
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
        subgraphDatas[graphName] = zeno::GraphData();
    }

    //zsg3.0以下的格式，子图将加入并成为项目的资产
    for (const auto& subgraph : graph.GetObject())
    {
        const std::string& graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;
        if (!_parseSubGraph(graphName,
                    subgraph.value,
                    nodesDescs,
                    subgraphDatas,
                    subgraphDatas[graphName]))
        {
            return false;
        }
    }

    zeno::GraphData mainData;
    if (doc.HasMember("main") || graph.HasMember("main"))
    {
        const rapidjson::Value& mainGraph = doc.HasMember("main") ? doc["main"] : graph["main"];
        if (!_parseSubGraph("/main", mainGraph, nodesDescs, subgraphDatas, mainData))
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
            const zeno::NodeDescs& descriptors,
            const zeno::AssetsData& subgraphDatas,
            zeno::GraphData& subgData)
{
    if (!subgraph.IsObject() || !subgraph.HasMember("nodes"))
        return false;

    //todo: should consider descript info. some info of outsock without connection show in descript info?

    const auto& nodes = subgraph["nodes"];
    if (nodes.IsNull())
        return false;

    /*目前还没有作用
    QMap<std::string, std::string> objIdToName;
    for (const auto &node : nodes.GetObject())
    {
        const std::string &nodeid = node.name.GetString();
        const auto &objValue = node.value;
        const std::string &name = objValue["name"].GetString();
        objIdToName[nodeid] = name;
    }
    */
    for (const auto& node : nodes.GetObject())
    {
        const std::string& nodeid = node.name.GetString();
        const zeno::NodeData& nodeData = _parseNode(graphPath, nodeid, node.value, descriptors, subgraphDatas, subgData.links);
        subgData.nodes.insert(std::make_pair(nodeid, nodeData));
    }
    return true;
}

zeno::NodeData Zsg2Reader::_parseNode(
                    const std::string& subgPath,
                    const std::string& nodeid,
                    const rapidjson::Value& nodeObj,
                    const zeno::NodeDescs& legacyDescs,
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

    //legacy case, should expand the subgraph node recursively.
    if (zeno::VER_3 != m_ioVer)
    {
        if (subgraphDatas.find(name) != subgraphDatas.end())
        {
            if (!isParsingAssets)
            {
                retNode.subgraph = zenoio::fork(subgPath + "/" + nodeid, subgraphDatas, name);
            }
        }
    }

    //if (!mgr.getSubgDesc(name, NODE_DESC()))
    //    initSockets(name, legacyDescs, retNode);

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(subgPath, nodeid, name, legacyDescs, objValue["inputs"], retNode, links);
    }
    if (objValue.HasMember("params"))
    {
        if (_parseParams2(nodeid, name, objValue["params"], retNode) == false)
            _parseParams(nodeid, name, objValue["params"], legacyDescs, retNode);
    }
    if (objValue.HasMember("outputs"))
    {
        _parseOutputs(nodeid, name, objValue["outputs"], retNode);
    }
    if (objValue.HasMember("customui-panel"))
    {
        _parseCustomPanel(nodeid, name, objValue["customui-panel"], retNode);
    }

    if (objValue.HasMember("uipos"))
    {
        auto uipos = objValue["uipos"].GetArray();
        QPointF pos = QPointF(uipos[0].GetFloat(), uipos[1].GetFloat());
        retNode.pos = pos;
    }
    if (objValue.HasMember("options"))
    {
        auto optionsArr = objValue["options"].GetArray();
        int opts = 0;
        for (int i = 0; i < optionsArr.Size(); i++)
        {
            ZASSERT_EXIT(optionsArr[i].IsString(), false);
            const std::string& optName = optionsArr[i].GetString();
            if (optName == "ONCE") {
                opts |= OPT_ONCE;
            } else if (optName == "PREP") {
                opts |= OPT_PREP;
            } else if (optName == "VIEW") {
                opts |= OPT_VIEW;
            } else if (optName == "MUTE") {
                opts |= OPT_MUTE;
            } else if (optName == "collapsed") {
                retNode.bCollasped = true;
            }
        }
        retNode.options = opts;
    }
    if (objValue.HasMember("dict_keys"))
    {
        _parseDictKeys(nodeid, objValue["dict_keys"], retNode);
    }
    if (objValue.HasMember("socket_keys"))
    {
        _parseBySocketKeys(nodeid, objValue, retNode);
    }
    if (objValue.HasMember("color_ramps"))
    {
        _parseColorRamps(nodeid, objValue["color_ramps"], retNode);
    }
    if (objValue.HasMember("points") && objValue.HasMember("handlers"))
    {
        _parseLegacyCurves(nodeid, objValue["points"], objValue["handlers"], retNode);
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
        BLACKBOARD_INFO blackboard;
        //use subkey "blackboard" for zeno2 io, but still compatible with zeno1
        const rapidjson::Value &blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

        if (blackBoardValue.HasMember("special")) {
            blackboard.special = blackBoardValue["special"].GetBool();
        }

        blackboard.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
        blackboard.content = blackBoardValue.HasMember("content") ? blackBoardValue["content"].GetString() : "";

        if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
            qreal w = blackBoardValue["width"].GetFloat();
            qreal h = blackBoardValue["height"].GetFloat();
            blackboard.sz = QSizeF(w, h);
        }
        if (blackBoardValue.HasMember("params")) {
            //todo
        }

        retNode.parmsNotDesc["blackboard"].name = "blackboard";
        retNode.parmsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
    }
    else if (name == "Group") 
    {
        BLACKBOARD_INFO blackboard;
        const rapidjson::Value &blackBoardValue = objValue.HasMember("blackboard") ? objValue["blackboard"] : objValue;

        blackboard.title = blackBoardValue.HasMember("title") ? blackBoardValue["title"].GetString() : "";
        blackboard.background = QColor(blackBoardValue.HasMember("background") ? blackBoardValue["background"].GetString() : "#3C4645");

        if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
            qreal w = blackBoardValue["width"].GetFloat();
            qreal h = blackBoardValue["height"].GetFloat();
            blackboard.sz = QSizeF(w, h);
        }
        if (blackBoardValue.HasMember("items")) {
            auto item_keys = blackBoardValue["items"].GetArray();
            for (int i = 0; i < item_keys.Size(); i++) {
                std::string key = item_keys[i].GetString();
                blackboard.items.append(key);
            }
        }

        retNode.parmsNotDesc["blackboard"].name = "blackboard";
        retNode.parmsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
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

void Zsg2Reader::initSockets(const std::string& name, const zeno::NodeDescs& legacyDescs, zeno::NodeData& ret)
{
    ZASSERT_EXIT(legacyDescs.find(name) != legacyDescs.end());
    const NODE_DESC& desc = legacyDescs[name];
    ret.inputs = desc.inputs;
    ret.params = desc.params;
    ret.outputs = desc.outputs;
    ret.parmsNotDesc = NodesMgr::initParamsNotDesc(name);
}

void Zsg2Reader::_parseViews(const rapidjson::Value& jsonViews, ZSG_PARSE_RESULT& res)
{
    if (jsonViews.HasMember("timeline"))
    {
        _parseTimeline(jsonViews["timeline"], res);
    }
}

void Zsg2Reader::_parseTimeline(const rapidjson::Value& jsonTimeline, ZSG_PARSE_RESULT& res)
{
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::start_frame) && jsonTimeline[timeline::start_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::end_frame) && jsonTimeline[timeline::end_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::curr_frame) && jsonTimeline[timeline::curr_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::always) && jsonTimeline[timeline::always].IsBool());

    res.timeline.beginFrame = jsonTimeline[timeline::start_frame].GetInt();
    res.timeline.endFrame = jsonTimeline[timeline::end_frame].GetInt();
    res.timeline.currFrame = jsonTimeline[timeline::curr_frame].GetInt();
    res.timeline.bAlways = jsonTimeline[timeline::always].GetBool();
}

void Zsg2Reader::_parseDictKeys(const std::string& id, const rapidjson::Value& objValue, zeno::NodeData& ret)
{
    ZASSERT_EXIT(objValue.HasMember("inputs") && objValue["inputs"].IsArray());
    auto input_keys = objValue["inputs"].GetArray();
    for (int i = 0; i < input_keys.Size(); i++)
    {
        std::string key = input_keys[i].GetString();
        ret.inputs[key].info.name = key;
        ret.inputs[key].info.nodeid = id;
        ret.inputs[key].info.control = CONTROL_NONE;
        ret.inputs[key].info.sockProp = SOCKPROP_EDITABLE;
        ret.inputs[key].info.type = "";
    }

    ZASSERT_EXIT(objValue.HasMember("outputs") && objValue["outputs"].IsArray());
    auto output_keys = objValue["outputs"].GetArray();
    for (int i = 0; i < output_keys.Size(); i++)
    {
        std::string key = output_keys[i].GetString();
        ret.outputs[key].info.name = key;
        ret.outputs[key].info.control = CONTROL_NONE;
        ret.outputs[key].info.sockProp = SOCKPROP_EDITABLE;
        ret.outputs[key].info.type = "";
    }
}

void Zsg2Reader::_parseBySocketKeys(const std::string& id, const rapidjson::Value& objValue, zeno::NodeData& ret)
{
    //deprecated.
    auto socket_keys = objValue["socket_keys"].GetArray();
    QStringList socketKeys;
    for (int i = 0; i < socket_keys.Size(); i++)
    {
        std::string key = socket_keys[i].GetString();
        socketKeys.append(key);
    }
}

zeno::zvariant Zsg2Reader::_parseDeflValue(
                const std::string &nodeCls,
                const zeno::NodeDescs &legacyDescs,
                const std::string& sockName,
                bool bInput,
                const rapidjson::Value &defaultValue)
{
    ZASSERT_EXIT(legacyDescs.find(nodeCls) != legacyDescs.end(), QVariant());
    NODE_DESC desc = legacyDescs[nodeCls];

    zeno::zvariant defl;
    if (cls == PARAM_INPUT)
    {
        if (!defaultValue.IsNull()) {
            SOCKET_INFO descInfo;
            if (desc.inputs.find(sockName) != desc.inputs.end()) {
                descInfo = desc.inputs[sockName].info;
            }
            defl = UiHelper::parseJsonByType(descInfo.type, defaultValue);
        }
    } else if (cls == PARAM_PARAM) {
        if (!defaultValue.IsNull()) {
            PARAM_INFO paramInfo;
            if (desc.params.find(sockName) != desc.params.end()) {
                paramInfo = desc.params[sockName];
            }
            //todo: need to consider SubInput/SubOutput?
            defl = UiHelper::parseJsonByType(paramInfo.typeDesc, defaultValue);
        }
    }
    return defl;
}

void Zsg2Reader::_parseInputs(
                const std::string& subgPath,
                const std::string& id,
                const std::string& nodeName,
                const zeno::NodeDescs& legacyDescs,
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
            INPUT_SOCKET socket;
            zeno::ParamInfo param;
            ret.inputs.insert(std::make_pair(inSock, param));
        }
        else if (inputObj.IsObject())
        {
            zeno::ParamInfo param = _parseSocket(subgPath, id, nodeName, inSock, true, inputObj, legacyDescs, links);
            ret.inputs.insert(std::make_pair(inSock, param));
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
        const zeno::NodeDescs& descriptors,
        zeno::LinksData& links)
{
    zeno::ParamInfo param;

    int sockprop = SOCKPROP_NORMAL;
    std::string sockProp;
    if (sockObj.HasMember("property"))
    {
        //ZASSERT_EXIT(sockObj["property"].IsString());
        sockProp = sockObj["property"].GetString();
    }

    zeno::ParamControl ctrl = zeno::ParamControl::Null;

    zeno::SocketProperty prop = zeno::SocketProperty::Normal;
    if (sockProp == "dict-panel")
        prop = zeno::SocketProperty::Normal;    //deprecated
    else if (sockProp == "editable")
        prop = zeno::SocketProperty::Editable;
    else if (sockProp == "group-line")
        prop = zeno::SocketProperty::Normal;    //deprecated

    param.prop = prop;
    param.name = sockName;

    if (m_bDiskReading &&
        (prop == zeno::SocketProperty::Editable ||
         nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
    {
        if (prop == zeno::SocketProperty::Editable) {
            //like extract dict.
            param.type = zeno::Param_String;
        } else {
            param.type = zeno::Param_Null;
        }
    }

    if (sockObj.HasMember("type") && sockObj.HasMember("default-value")) {
        param.type = zeno::convertToType(sockObj["type"].GetString());
        param.defl = zeno::jsonValueToZVar(sockObj["default-value"], param.type);
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
        _parseDictPanel(subgPath, bInput, sockObj["dictlist-panel"], id, sockName, nodeCls, ret, links);
    }
    if (sockObj.HasMember("control") && 
        (descriptors.find(nodeCls) == descriptors.end() || 
            !descriptors[nodeCls].inputs.contains(socket.name) ||
            GraphsManagment::instance().getSubgDesc(nodeCls, NODE_DESC())))
	{
        PARAM_CONTROL ctrl;
        QVariant props;
        bool bret = JsonHelper::importControl(sockObj["control"], ctrl, props);
        if (bret){
            socket.control = ctrl;
            socket.ctrlProps = props.toMap();
        }
    }

    if (sockObj.HasMember("tooltip")) 
    {
        param.tooltip = sockObj["tooltip"].GetString();
    }
}

zeno::NodesData Zsg2Reader::_parseChildren(const rapidjson::Value& jsonNodes)
{
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
        bool bCollasped = dictPanelObj["collasped"].GetBool();
        if (bInput) {
            if (ret.inputs.find(sockName) != ret.inputs.end()) {
                ret.inputs[sockName].info.dictpanel.bCollasped = bCollasped;
            }
        } else {
            if (ret.outputs.find(sockName) != ret.outputs.end()) {
                ret.outputs[sockName].info.dictpanel.bCollasped = bCollasped;
            }
        }
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
                link = std::string::fromUtf8(inputObj["link"].GetString());
            }

            //standard inputs desc by latest descriptors.
            if (ret.inputs.find(sockName) != ret.inputs.end())
            {
                INPUT_SOCKET &inSocket = ret.inputs[sockName];
                DICTKEY_INFO item;
                item.key = keyName;

                std::string outSockPath = link;
                QStringList lst = outSockPath.split(cPathSeperator, QtSkipEmptyParts);
                if (lst.size() > 2)
                    outSockPath = UiHelper::constructObjPath(lst[0], lst[1], lst[2]);
                if (!outSockPath.isEmpty())
                {
                    std::string inSockPath = std::string("%1/%2:[node]/inputs/%3/%4").arg(subgPath).arg(id).arg(sockName).arg(keyName);
                    EdgeInfo edge(outSockPath, inSockPath);
                    links.append(edge);
                }
                inSocket.info.dictpanel.keys.append(item);
            }
            if (ret.outputs.find(sockName) != ret.outputs.end())
            {
                OUTPUT_SOCKET &outSocket = ret.outputs[sockName];
                DICTKEY_INFO item;
                item.key = keyName;

                std::string newKeyPath = "[node]/outputs/" + sockName + "/" + keyName;
                outSocket.info.dictpanel.keys.append(item);
                //no need to import link here.
            }
        }
    }
}

void Zsg2Reader::_parseOutputs(const std::string &id, const std::string &nodeName, const rapidjson::Value& outputs, zeno::NodeData& ret)
{
    for (const auto& outObj : outputs.GetObject())
    {
        const std::string& outSock = outObj.name.GetString();
        if (ret.outputs.find(outSock) == ret.outputs.end()) {
            ret.outputs[outSock] = OUTPUT_SOCKET();
            ret.outputs[outSock].info.name = outSock;
        }
        const auto& sockObj = outObj.value;
        if (sockObj.IsObject())
        {
            if (sockObj.HasMember("dictlist-panel")) {
                _parseDictPanel("", false, sockObj["dictlist-panel"], id, outSock, nodeName, ret, zeno::LinksData());
            }
            if (sockObj.HasMember("tooltip")) {
                ret.outputs[outSock].info.toolTip = std::string::fromUtf8(sockObj["tooltip"].GetString());
            }
        }
    }
}

void Zsg2Reader::_parseCustomPanel(const std::string& id, const std::string& nodeName, const rapidjson::Value& jsonCutomUI, zeno::NodeData& ret)
{
    VPARAM_INFO invisibleRoot = zenomodel::importCustomUI(jsonCutomUI);
    ret.customPanel = invisibleRoot;
}

void Zsg2Reader::_parseColorRamps(const std::string& id, const rapidjson::Value& jsonColorRamps, zeno::NodeData& ret)
{
    //deprecated
    if (jsonColorRamps.IsNull())
        return;

    COLOR_RAMPS colorRamps;
    RAPIDJSON_ASSERT(jsonColorRamps.IsArray());
    const auto& arr = jsonColorRamps.GetArray();
    for (int i = 0; i < arr.Size(); i++)
    {
        const auto& colorRampObj = arr[i];
        RAPIDJSON_ASSERT(colorRampObj.IsArray());
        const auto &rampArr = colorRampObj.GetArray();
        const auto &rgb = rampArr[1].GetArray();

        COLOR_RAMP clrRamp;
        clrRamp.pos = rampArr[0].GetFloat();
        clrRamp.r = rgb[0].GetFloat();
        clrRamp.g = rgb[1].GetFloat();
        clrRamp.b = rgb[2].GetFloat();
        colorRamps.push_back(clrRamp);
    }
}

void Zsg2Reader::_parseLegacyCurves(const std::string& id,
                                   const rapidjson::Value& jsonPoints,
                                   const rapidjson::Value& jsonHandlers,
                                   zeno::NodeData& ret)
{
    //deprecated
    if (jsonPoints.IsNull() || jsonHandlers.IsNull())
        return;

    QVector<QPointF> pts;
    RAPIDJSON_ASSERT(jsonPoints.IsArray());
    const auto &arr = jsonPoints.GetArray();
    for (int i = 0; i < arr.Size(); i++)
    {
        const auto &pointObj = arr[i];
        bool bSucceed = false;
        QPointF pt = UiHelper::parsePoint(pointObj, bSucceed);
        ZASSERT_EXIT(bSucceed);
        pts.append(pt);
    }

    RAPIDJSON_ASSERT(jsonHandlers.IsArray());
    const auto &arr2 = jsonHandlers.GetArray();
    QVector<QPair<QPointF, QPointF>> hdls;
    for (int i = 0; i < arr2.Size(); i++)
    {
        RAPIDJSON_ASSERT(arr2[i].IsArray() && arr2[i].Size() == 2);
        const auto &arr_ = arr2[i].GetArray();

        bool bSucceed = false;
        QPointF leftHdl = UiHelper::parsePoint(arr_[0], bSucceed);
        ZASSERT_EXIT(bSucceed);
        QPointF rightHdl = UiHelper::parsePoint(arr_[1], bSucceed);
        ZASSERT_EXIT(bSucceed);

        hdls.append(QPair(leftHdl, rightHdl));
    }
}

zeno::NodeDescs Zsg2Reader::_parseDescs(const rapidjson::Value& jsonDescs)
{
    auto& mgr = GraphsManagment::instance();
    zeno::NodeDescs _descs = mgr.descriptors();
    for (const auto& node : jsonDescs.GetObject())
    {
        const std::string& nodeCls = node.name.GetString();
        const auto& objValue = node.value;

        if (_descs.find(nodeCls) != _descs.end() && !mgr.getSubgDesc(nodeCls, NODE_DESC()))
        {
            continue;
        }

        NODE_DESC desc;
        desc.name = nodeCls;
        if (objValue.HasMember("inputs"))
        {
            if (objValue["inputs"].IsArray()) 
            {
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

                        //zeno::log_info("input_triple[2] = {}", input_triple[2].GetType());
                        //Q_ASSERT(!socketName.isEmpty());
                        if (!socketName.isEmpty())
                        {
                            CONTROL_INFO infos = UiHelper::getControlByType(nodeCls, PARAM_INPUT, socketName, socketType);

                            INPUT_SOCKET inputSocket;
                            inputSocket.info = SOCKET_INFO("", socketName);
                            inputSocket.info.type = socketType;
                            inputSocket.info.control = infos.control;
                            inputSocket.info.ctrlProps = infos.controlProps.toMap();
                            inputSocket.info.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                            desc.inputs.insert(socketName, inputSocket);
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
                    QVariant var = JsonHelper::importDescriptor(input.value, socketName,PARAM_INPUT);
                    if (var.canConvert<INPUT_SOCKET>()) 
                    {
                        desc.inputs.insert(socketName, var.value<INPUT_SOCKET>());
                    }
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

                        //zeno::log_info("param_triple[2] = {}", param_triple[2].GetType());
                        //Q_ASSERT(!socketName.isEmpty());
                        if (!socketName.isEmpty())
                        {
                            CONTROL_INFO infos = UiHelper::getControlByType(nodeCls, PARAM_PARAM, socketName, socketType);
                            PARAM_INFO paramInfo;
                            paramInfo.bEnableConnect = false;
                            paramInfo.control = infos.control;
                            paramInfo.controlProps = infos.controlProps.toMap();
                            paramInfo.name = socketName;
                            paramInfo.typeDesc = socketType;
                            paramInfo.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                            desc.params.insert(socketName, paramInfo);
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
                    QVariant var = JsonHelper::importDescriptor(param.value, socketName, PARAM_PARAM);
                    if (var.canConvert<PARAM_INFO>()) 
                    {
                        desc.params.insert(socketName, var.value<PARAM_INFO>());
                    }
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

                        //Q_ASSERT(!socketName.isEmpty());
                        if (!socketName.isEmpty())
                        {
                            OUTPUT_SOCKET outputSocket;
                            outputSocket.info = SOCKET_INFO("", socketName);
                            outputSocket.info.type = socketType;
                            outputSocket.info.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                            desc.outputs.insert(socketName, outputSocket);
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
                    QVariant var = JsonHelper::importDescriptor(output.value, socketName, PARAM_OUTPUT);
                    if (var.canConvert<OUTPUT_SOCKET>()) 
                    {
                        desc.outputs.insert(socketName, var.value<OUTPUT_SOCKET>());
                    }
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

        _descs.insert(nodeCls, desc);
    }
    return _descs;
}

void Zsg2Reader::_parseParams(
            const std::string& id,
            const std::string& nodeCls,
            const rapidjson::Value& jsonParams,
            const zeno::NodeDescs& legacyDescs,
            zeno::NodeData& ret)
{
    if (jsonParams.IsObject())
    {
        for (const auto& paramObj : jsonParams.GetObject())
        {
            const std::string& name = paramObj.name.GetString();
            const rapidjson::Value& val = paramObj.value;

            ZASSERT_EXIT(legacyDescs.find(nodeCls) != legacyDescs.end());
            NODE_DESC desc = legacyDescs[nodeCls];
            QVariant var;
            if (!val.IsNull()) {
                PARAM_INFO paramInfo;
                if (desc.params.find(name) != desc.params.end()) {
                    paramInfo = desc.params[name];
                }
                if (nodeCls == "SubInput" || nodeCls == "SubOutput")
                    var = UiHelper::parseJsonByValue(paramInfo.typeDesc, val); //dynamic type on SubInput defl.
                else
                    var = UiHelper::parseJsonByType(paramInfo.typeDesc, val);
            }
            if (ret.params.find(name) != ret.params.end())
            {
                zeno::log_trace("found param name {}", name.toStdString());
                ret.params[name].value = var;
            } else {
                ret.parmsNotDesc[name].value = var;

                if (name == "_KEYS" && (nodeCls == "MakeDict" || nodeCls == "ExtractDict" || nodeCls == "MakeList")) {
                    //parse by socket_keys in zeno2.
                    return;
                }
                if (nodeCls == "MakeCurvemap" && (name == "_POINTS" || name == "_HANDLERS")) {
                    PARAM_INFO paramData;
                    paramData.control = CONTROL_NONVISIBLE;
                    paramData.name = name;
                    paramData.bEnableConnect = false;
                    paramData.value = var;
                    ret.params[name] = paramData;
                    return;
                }
                if (nodeCls == "MakeHeatmap" && name == "_RAMPS") {
                    PARAM_INFO paramData;
                    paramData.control = CONTROL_COLOR;
                    paramData.name = name;
                    paramData.bEnableConnect = false;
                    paramData.value = var;
                    ret.params[name] = paramData;
                    return;
                }
                if (nodeCls == "DynamicNumber" && (name == "_CONTROL_POINTS" || name == "_TMP")) {
                    PARAM_INFO paramData;
                    paramData.control = CONTROL_NONVISIBLE;
                    paramData.name = name;
                    paramData.bEnableConnect = false;
                    paramData.value = var;
                    ret.params[name] = paramData;
                    return;
                }
                zeno::log_warn("not found param name {}", name.toStdString());
            }
        }

        if (nodeCls == "SubInput" || nodeCls == "SubOutput") {
            ZASSERT_EXIT(ret.params.find("name") != ret.params.end() &&
                         ret.params.find("type") != ret.params.end() &&
                         ret.params.find("defl") != ret.params.end());

            const std::string &descType = ret.params["type"].value.toString();
            PARAM_INFO &defl = ret.params["defl"];
            defl.control = UiHelper::getControlByType(descType);
            defl.value = UiHelper::parseVarByType(descType, defl.value);
            defl.typeDesc = descType;
        }
    } else {
        if (nodeCls == "Blackboard" && jsonParams.IsArray())
        {
            //deprecate by zeno-old.
            return;
        }
        zeno::log_warn("not object json param");
    }
}

bool Zsg2Reader::_parseParams2(const std::string& id, const std::string &nodeCls, const rapidjson::Value &jsonParams, zeno::NodeData& ret) 
{
    if (jsonParams.IsObject()) {
        //PARAMS_INFO params;
        for (const auto &paramObj : jsonParams.GetObject()) {
            const std::string &name = paramObj.name.GetString();
            const rapidjson::Value &value = paramObj.value;
            if (!value.IsObject() || !value.HasMember(iotags::params::params_valueKey)) //compatible old version
                return false;

            PARAM_INFO paramData;
            if (value.HasMember("type"))
                paramData.typeDesc = value["type"].GetString();
            QVariant var;
            if (nodeCls == "SubInput" || nodeCls == "SubOutput")
                var = UiHelper::parseJsonByValue(paramData.typeDesc, value[iotags::params::params_valueKey]); //dynamic type on SubInput defl.
            else
                var = UiHelper::parseJsonByType(paramData.typeDesc, value[iotags::params::params_valueKey]);

            CONTROL_INFO ctrlInfo = UiHelper::getControlByType(nodeCls, PARAM_PARAM, name, paramData.typeDesc);
            if (ctrlInfo.control != CONTROL_NONE && ctrlInfo.controlProps.isValid()) {
                paramData.control = ctrlInfo.control;
                paramData.controlProps = ctrlInfo.controlProps;
            }
            else if (value.HasMember("control")) {
                PARAM_CONTROL ctrl;
                QVariant props;
                bool bret = JsonHelper::importControl(value["control"], ctrl, props);
                if (bret) {
                    paramData.control = ctrl;
                    paramData.controlProps = props;
                }
            }
            if (value.HasMember("tooltip")) 
            {
                std::string toolTip = std::string::fromUtf8(value["tooltip"].GetString());
                paramData.toolTip = toolTip;
            }
            paramData.name = name;
            paramData.bEnableConnect = false;
            paramData.value = var;
            ret.params[name] = paramData;
        }
    }
    return true;
}

}
