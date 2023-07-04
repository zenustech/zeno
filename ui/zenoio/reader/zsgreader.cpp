#include "zsgreader.h"
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

using namespace zeno::iotags;
using namespace zeno::iotags::curve;

namespace zenoio {

ZsgReader::ZsgReader() : m_bDiskReading(true), m_ioVer(VER_3) {}

ZsgReader& ZsgReader::getInstance()
{
    static ZsgReader reader;
    return reader;
}

bool ZsgReader::importNodes(
            IGraphsModel* pModel,
            const QModelIndex& subgIdx,
            const QString& nodeJson,
            const QPointF& targetPos,
            SUBGRAPH_DATA& subgraph)
{
    m_bDiskReading = false;
    rapidjson::Document doc;
    QByteArray bytes = nodeJson.toUtf8();
    doc.Parse(bytes);

    if (!doc.IsObject() || !doc.HasMember("nodes"))
        return false;

    const rapidjson::Value& nodes = doc["nodes"];
    if (nodes.IsNull())
        return false;

    QString subgPath = subgIdx.data(ROLE_OBJPATH).toString();

    QStringList idents;
    NODES_DATA nodeDatas;
    NODE_DESCS descs = GraphsManagment::instance().descriptors();
    for (const auto &node : nodes.GetObject())
    {
        const QString &nodeid = node.name.GetString();
        idents.append(nodeid);
        NODE_DATA& nodeData = subgraph.nodes[nodeid];
        if (!_parseNode(subgPath, nodeid, node.value, descs, QMap<QString, SUBGRAPH_DATA>(), nodeData,
            subgraph.links))
        {
            return false;
        }
    }
    return true;
}

bool ZsgReader::openFile(const QString& fn, ZSG_PARSE_RESULT& result)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("cannot open zsg file: {} ({})", fn.toStdString(),
                       file.errorString().toStdString());
        return false;
    }

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    m_ioVer = zenoio::VER_3;
    if (doc.HasMember("version"))
    {
        ZASSERT_EXIT(doc["version"].IsString(), false);
        QString ver = doc["version"].GetString();
        if (ver == "v2")
            m_ioVer = zenoio::VER_2;
        else if (ver == "v2.5")
            m_ioVer = zenoio::VER_2_5;
        else if (ver == "v3.0")
            m_ioVer = zenoio::VER_3;
    }

    if (!doc.IsObject())
    {
        zeno::log_error("");
        return false;
    }

    if ((m_ioVer != zenoio::VER_3 && !doc.HasMember("graph")) ||
        (m_ioVer == zenoio::VER_3 && (!doc.HasMember("main") || !doc.HasMember("subgraphs"))))
    {
        return false;
    }

    const rapidjson::Value &graph = doc.HasMember("graph") ? doc["graph"] : doc["subgraphs"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}", fn.toStdString());
        return false;
    }

    ZASSERT_EXIT(doc.HasMember("descs"), false);
    NODE_DESCS nodesDescs = _parseDescs(doc["descs"]);
    if (!ret) {
        return false;
    }

    QMap<QString, SUBGRAPH_DATA> subgraphDatas;
    //init keys
    for (const auto& subgraph : graph.GetObject())
    {
        const QString &graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;
        subgraphDatas[graphName] = SUBGRAPH_DATA();
    }

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
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

    SUBGRAPH_DATA mainData;
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

    result.ver = m_ioVer;
    result.descs = nodesDescs;
    result.mainGraph = mainData;
    result.subgraphs = subgraphDatas;
    return true;
}

bool zenoio::ZsgReader::openSubgraphFile(const QString& fn, ZSG_PARSE_RESULT& result)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("cannot open zsg file: {} ({})", fn.toStdString(),
            file.errorString().toStdString());
        return false;
    }

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    if (!doc.IsObject())
    {
        zeno::log_error("");
        return false;
    }
    if (!doc.HasMember("subgraphs") && !doc.HasMember("graph"))
        return false;
    const rapidjson::Value& graph = doc.HasMember("subgraphs")? doc["subgraphs"] : doc["graph"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}", fn.toStdString());
        return false;
    }

    ZASSERT_EXIT(doc.HasMember("descs"), false);
    NODE_DESCS nodesDescs = _parseDescs(doc["descs"]);
    if (!ret) {
        return false;
    }

    QMap<QString, SUBGRAPH_DATA> subgraphDatas;
    //init keys
    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        if ("main" == graphName)
            continue;
        subgraphDatas[graphName] = SUBGRAPH_DATA();
    }

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
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

    result.descs = nodesDescs;
    result.subgraphs = subgraphDatas;
    return true;
}

bool ZsgReader::_parseSubGraph(
            const QString& subgPath,
            const rapidjson::Value& subgraph,
            const NODE_DESCS& descriptors,
            const QMap<QString, SUBGRAPH_DATA>& subgraphDatas,
            SUBGRAPH_DATA& subgData)
{
    if (!subgraph.IsObject() || !subgraph.HasMember("nodes"))
        return false;

    //todo: should consider descript info. some info of outsock without connection show in descript info?

    const auto& nodes = subgraph["nodes"];
    if (nodes.IsNull())
        return false;

    QMap<QString, QString> objIdToName;
    for (const auto &node : nodes.GetObject())
    {
        const QString &nodeid = node.name.GetString();
        const auto &objValue = node.value;
        const QString &name = objValue["name"].GetString();
        objIdToName[nodeid] = name;
    }

    for (const auto& node : nodes.GetObject())
    {
        const QString& nodeid = node.name.GetString();
        NODE_DATA& nodeData = subgData.nodes[nodeid];
        _parseNode(subgPath, nodeid, node.value, descriptors, subgraphDatas, nodeData, subgData.links);
    }

    return true;
}

bool ZsgReader::_parseNode(
                    const QString& subgPath,
                    const QString& nodeid,
                    const rapidjson::Value& nodeObj,
                    const NODE_DESCS& legacyDescs,
                    const QMap<QString, SUBGRAPH_DATA>& subgraphDatas,
                    NODE_DATA& ret,
                    LINKS_DATA& links)
{
    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const QString& name = nameValue.GetString();

    ret.ident = nodeid;
    ret.nodeCls = name;
    auto &mgr = GraphsManagment::instance();
    ret.type = mgr.nodeType(name);
    if (subgraphDatas.contains(name))
        ret.type = SUBGRAPH_NODE;

    QString customName;
    if (objValue.HasMember("customName")) {
        const QString &tmp = objValue["customName"].GetString();
        customName = tmp;
    }
    ret.customName = customName;

    //legacy case, should expand the subgraph node recursively.
    if (zenoio::VER_3 != m_ioVer)
    {
        if (subgraphDatas.find(name) != subgraphDatas.end())
        {
            ret.type = SUBGRAPH_NODE;
            if (subgPath.startsWith("/main"))
            {
                ret.children = UiHelper::fork(subgPath + "/" + nodeid, subgraphDatas, name, links);
            }
        }
    }

    initSockets(name, legacyDescs, ret);

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(subgPath, nodeid, name, legacyDescs, objValue["inputs"], ret, links);
    }
    if (objValue.HasMember("params"))
    {
        if (_parseParams2(nodeid, name, objValue["params"], ret) == false)
            _parseParams(nodeid, name, objValue["params"], legacyDescs, ret);
    }
    if (objValue.HasMember("outputs"))
    {
        _parseOutputs(nodeid, name, objValue["outputs"], ret);
    }
    if (objValue.HasMember("customui-panel"))
    {
        _parseCustomPanel(nodeid, name, objValue["customui-panel"], ret);
    }

    if (objValue.HasMember("uipos"))
    {
        auto uipos = objValue["uipos"].GetArray();
        QPointF pos = QPointF(uipos[0].GetFloat(), uipos[1].GetFloat());
        ret.pos = pos;
    }
    if (objValue.HasMember("options"))
    {
        auto optionsArr = objValue["options"].GetArray();
        int opts = 0;
        for (int i = 0; i < optionsArr.Size(); i++)
        {
            ZASSERT_EXIT(optionsArr[i].IsString(), false);
            const QString& optName = optionsArr[i].GetString();
            if (optName == "ONCE") {
                opts |= OPT_ONCE;
            } else if (optName == "PREP") {
                opts |= OPT_PREP;
            } else if (optName == "VIEW") {
                opts |= OPT_VIEW;
            } else if (optName == "MUTE") {
                opts |= OPT_MUTE;
            } else if (optName == "collapsed") {
                ret.bCollasped = true;
            }
        }
        ret.options = opts;
    }
    if (objValue.HasMember("dict_keys"))
    {
        _parseDictKeys(nodeid, objValue["dict_keys"], ret);
    }
    if (objValue.HasMember("socket_keys"))
    {
        _parseBySocketKeys(nodeid, objValue, ret);
    }
    if (objValue.HasMember("color_ramps"))
    {
        _parseColorRamps(nodeid, objValue["color_ramps"], ret);
    }
    if (objValue.HasMember("points") && objValue.HasMember("handlers"))
    {
        _parseLegacyCurves(nodeid, objValue["points"], objValue["handlers"], ret);
    }
    if (objValue.HasMember("children"))
    {
        SUBGRAPH_DATA subg;
        _parseSubGraph(subgPath + '/' + nodeid, objValue["children"], legacyDescs, subgraphDatas, subg);
        ret.children = subg.nodes;
        links.append(subg.links);
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

        ret.parmsNotDesc["blackboard"].name = "blackboard";
        ret.parmsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
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
                QString key = item_keys[i].GetString();
                blackboard.items.append(key);
            }
        }

        ret.parmsNotDesc["blackboard"].name = "blackboard";
        ret.parmsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
    }

    return true;
}

void ZsgReader::_parseChildNodes(
                    const QString& rootPath,
                    const rapidjson::Value& jsonNodes,
                    const NODE_DESCS& descriptors,
                    NODE_DATA& ret)
{
    if (!jsonNodes.HasMember("nodes"))
        return;
}

void ZsgReader::initSockets(const QString& name, const NODE_DESCS& legacyDescs, NODE_DATA& ret)
{
    ZASSERT_EXIT(legacyDescs.find(name) != legacyDescs.end());
    const NODE_DESC& desc = legacyDescs[name];
    ret.inputs = desc.inputs;
    ret.params = desc.params;
    ret.outputs = desc.outputs;
    ret.parmsNotDesc = NodesMgr::initParamsNotDesc(name);
}

void ZsgReader::_parseViews(const rapidjson::Value& jsonViews, ZSG_PARSE_RESULT& res)
{
    if (jsonViews.HasMember("timeline"))
    {
        _parseTimeline(jsonViews["timeline"], res);
    }
}

void ZsgReader::_parseTimeline(const rapidjson::Value& jsonTimeline, ZSG_PARSE_RESULT& res)
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

void ZsgReader::_parseDictKeys(const QString& id, const rapidjson::Value& objValue, NODE_DATA& ret)
{
    ZASSERT_EXIT(objValue.HasMember("inputs") && objValue["inputs"].IsArray());
    auto input_keys = objValue["inputs"].GetArray();
    for (int i = 0; i < input_keys.Size(); i++)
    {
        QString key = input_keys[i].GetString();
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
        QString key = output_keys[i].GetString();
        ret.outputs[key].info.name = key;
        ret.outputs[key].info.control = CONTROL_NONE;
        ret.outputs[key].info.sockProp = SOCKPROP_EDITABLE;
        ret.outputs[key].info.type = "";
    }
}

void ZsgReader::_parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, NODE_DATA& ret)
{
    //deprecated.
    auto socket_keys = objValue["socket_keys"].GetArray();
    QStringList socketKeys;
    for (int i = 0; i < socket_keys.Size(); i++)
    {
        QString key = socket_keys[i].GetString();
        socketKeys.append(key);
    }
}

QVariant ZsgReader::_parseDeflValue(
                const QString &nodeCls,
                const NODE_DESCS &legacyDescs,
                const QString& sockName,
                PARAM_CLASS cls,
                const rapidjson::Value &defaultValue)
{
    ZASSERT_EXIT(legacyDescs.find(nodeCls) != legacyDescs.end(), QVariant());
    NODE_DESC desc = legacyDescs[nodeCls];

    QVariant defl;
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

void ZsgReader::_parseInputs(
                const QString& subgPath,
                const QString& id,
                const QString& nodeName,
                const NODE_DESCS& legacyDescs,
                const rapidjson::Value& inputs,
                NODE_DATA& ret,
                LINKS_DATA& links)
{
    for (const auto& inObj : inputs.GetObject())
    {
        const QString& inSock = inObj.name.GetString();
        const auto& inputObj = inObj.value;
        if (inputObj.IsArray())
        {
            //legacy io format, like [xxx-node, xxx-socket, defl]
            const auto& arr = inputObj.GetArray();
            ZASSERT_EXIT(arr.Size() >= 2 && arr.Size() <= 3);

            QString outId, outSock;
            int n = arr.Size();
            ZASSERT_EXIT(n == 3);

            INPUT_SOCKET socket;
            if (ret.inputs.contains(inSock))
                socket = ret.inputs[inSock];
            socket.info.defaultValue = _parseDeflValue(nodeName, legacyDescs, inSock, PARAM_INPUT, arr[2]);

            if (arr[0].IsString() && arr[1].IsString())
            {
                outId = arr[0].GetString();
                outSock = arr[1].GetString();
                QString outLinkPath = QString("%1/%2:[node]/outputs/%3").arg(subgPath).arg(outId).arg(outSock);
                QString inLinkPath = QString("%1/%2:[node]/inputs/%3").arg(subgPath).arg(id).arg(inSock);
                links.append(EdgeInfo(outLinkPath, inLinkPath));
            }

            ret.inputs[inSock] = socket;
        }
        else if (inputObj.IsNull())
        {
            INPUT_SOCKET socket;
            ret.inputs.insert(inSock, socket);
        }
        else if (inputObj.IsObject())
        {
            _parseSocket(subgPath, id, nodeName, inSock, true, inputObj, legacyDescs, ret, links);
        }
        else
        {
            Q_ASSERT(false);
        }
    }
}

void ZsgReader::_parseSocket(
        const QString& subgPath,
        const QString& id,
        const QString& nodeCls,
        const QString& sockName,
        bool bInput,
        const rapidjson::Value& sockObj,
        const NODE_DESCS& descriptors,
        NODE_DATA& ret,
        LINKS_DATA& links)
{
    int sockprop = SOCKPROP_NORMAL;
    QString sockProp;
    if (sockObj.HasMember("property"))
    {
        ZASSERT_EXIT(sockObj["property"].IsString());
        sockProp = QString::fromUtf8(sockObj["property"].GetString());
    }

    PARAM_CONTROL ctrl = CONTROL_NONE;
    SOCKET_PROPERTY prop = SOCKPROP_NORMAL;
    if (sockProp == "dict-panel")
        prop = SOCKPROP_DICTLIST_PANEL;
    else if (sockProp == "editable")
        prop = SOCKPROP_EDITABLE;
    else if (sockProp == "group-line")
        prop = SOCKPROP_GROUP_LINE;

    SOCKET_INFO& socket = bInput ? ret.inputs[sockName].info : ret.outputs[sockName].info;
    socket.sockProp = prop;
    socket.name = sockName;

    if (m_bDiskReading &&
        (prop == SOCKPROP_EDITABLE || nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
    {
        if (prop == SOCKPROP_EDITABLE) {
            //like extract dict.
            socket.type = "string";
        } else {
            socket.type = "";
        }
    }

    PARAM_CLASS cls = bInput ? PARAM_INPUT : PARAM_OUTPUT;

    if (sockObj.HasMember("type")) {
        socket.type = sockObj["type"].GetString();
    }
    if (sockObj.HasMember("default-value"))
    {
        if (descriptors.find(sockName) != descriptors.end())
            socket.defaultValue = _parseDeflValue(nodeCls, descriptors, sockName, cls, sockObj["default-value"]);
        else if (!socket.type.isEmpty() && !sockObj["default-value"].IsNull()) {
            socket.defaultValue = UiHelper::parseJsonByType(socket.type, sockObj["default-value"]);
        }
    }

    //link:
    if (bInput && sockObj.HasMember("link") && sockObj["link"].IsString())
    {
        QString outLinkPath = QString::fromUtf8(sockObj["link"].GetString());
        QStringList lst = outLinkPath.split(cPathSeperator, QtSkipEmptyParts);
        if (lst.size() > 2)
            outLinkPath = UiHelper::constructObjPath(lst[0], lst[1], lst[2]);
        QString inLinkPath = QString("%1/%2:[node]/inputs/%3").arg(subgPath).arg(id).arg(sockName);
        EdgeInfo fullLink(outLinkPath, inLinkPath);
        links.append(fullLink);
    }

    if (sockObj.HasMember("dictlist-panel"))
    {
        _parseDictPanel(subgPath, bInput, sockObj["dictlist-panel"], id, sockName, nodeCls, ret, links);
    }
    if (sockObj.HasMember("control") && descriptors.find(nodeCls) == descriptors.end()) 
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
        socket.toolTip = QString::fromUtf8(sockObj["tooltip"].GetString());
    }
}

NODES_DATA ZsgReader::_parseChildren(const rapidjson::Value& jsonNodes)
{
    NODES_DATA children;
    //_parseSubGraph(, , , children);
    return children;
}

void ZsgReader::_parseDictPanel(
            const QString& subgPath,
            bool bInput,
            const rapidjson::Value& dictPanelObj, 
            const QString& id,
            const QString& sockName,
            const QString& nodeName,
            NODE_DATA& ret,
            LINKS_DATA& links)
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
            const QString& keyName = kv.name.GetString();
            const rapidjson::Value& inputObj = kv.value;

            QString link;
            if (inputObj.HasMember("link") && inputObj["link"].IsString())
            {
                link = QString::fromUtf8(inputObj["link"].GetString());
            }

            //standard inputs desc by latest descriptors.
            if (ret.inputs.find(sockName) != ret.inputs.end())
            {
                INPUT_SOCKET &inSocket = ret.inputs[sockName];
                DICTKEY_INFO item;
                item.key = keyName;

                QString outSockPath = link;
                QStringList lst = outSockPath.split(cPathSeperator, QtSkipEmptyParts);
                if (lst.size() > 2)
                    outSockPath = UiHelper::constructObjPath(lst[0], lst[1], lst[2]);
                if (!outSockPath.isEmpty())
                {
                    QString inSockPath = QString("%1/%2:[node]/inputs/%3/%4").arg(subgPath).arg(id).arg(sockName).arg(keyName);
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

                QString newKeyPath = "[node]/outputs/" + sockName + "/" + keyName;
                outSocket.info.dictpanel.keys.append(item);
                //no need to import link here.
            }
        }
    }
}

void ZsgReader::_parseOutputs(const QString &id, const QString &nodeName, const rapidjson::Value& outputs, NODE_DATA& ret)
{
    for (const auto& outObj : outputs.GetObject())
    {
        const QString& outSock = outObj.name.GetString();
        if (ret.outputs.find(outSock) == ret.outputs.end()) {
            ret.outputs[outSock] = OUTPUT_SOCKET();
            ret.outputs[outSock].info.name = outSock;
        }
        const auto& sockObj = outObj.value;
        if (sockObj.IsObject())
        {
            if (sockObj.HasMember("dictlist-panel")) {
                _parseDictPanel("", false, sockObj["dictlist-panel"], id, outSock, nodeName, ret, LINKS_DATA());
            }
            if (sockObj.HasMember("tooltip")) {
                ret.outputs[outSock].info.toolTip = QString::fromUtf8(sockObj["tooltip"].GetString());
            }
        }
    }
}

void ZsgReader::_parseCustomPanel(const QString& id, const QString& nodeName, const rapidjson::Value& jsonCutomUI, NODE_DATA& ret)
{
    VPARAM_INFO invisibleRoot = zenomodel::importCustomUI(jsonCutomUI);
    ret.customPanel = invisibleRoot;
}

void ZsgReader::_parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, NODE_DATA& ret)
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

void ZsgReader::_parseLegacyCurves(const QString& id,
                                   const rapidjson::Value& jsonPoints,
                                   const rapidjson::Value& jsonHandlers,
                                   NODE_DATA& ret)
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

NODE_DESCS ZsgReader::_parseDescs(const rapidjson::Value& jsonDescs)
{
    auto& mgr = GraphsManagment::instance();
    NODE_DESCS _descs = mgr.descriptors();
    for (const auto& node : jsonDescs.GetObject())
    {
        const QString& nodeCls = node.name.GetString();
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
                        QString socketType, socketName, socketDefl;
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
                    QString socketName = input.name.GetString();
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
                        QString socketType, socketName, socketDefl;

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
                    QString socketName = param.name.GetString();
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
                        QString socketType, socketName, socketDefl;

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
                    QString socketName = output.name.GetString();
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

void ZsgReader::_parseParams(
            const QString& id,
            const QString& nodeCls,
            const rapidjson::Value& jsonParams,
            const NODE_DESCS& legacyDescs,
            NODE_DATA& ret)
{
    if (jsonParams.IsObject())
    {
        for (const auto& paramObj : jsonParams.GetObject())
        {
            const QString& name = paramObj.name.GetString();
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

            const QString &descType = ret.params["type"].value.toString();
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

bool ZsgReader::_parseParams2(const QString& id, const QString &nodeCls, const rapidjson::Value &jsonParams, NODE_DATA& ret) 
{
    if (jsonParams.IsObject()) {
        //PARAMS_INFO params;
        for (const auto &paramObj : jsonParams.GetObject()) {
            const QString &name = paramObj.name.GetString();
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
                QString toolTip = QString::fromUtf8(value["tooltip"].GetString());
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
