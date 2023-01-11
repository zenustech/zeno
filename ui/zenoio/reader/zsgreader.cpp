#include "zsgreader.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/customui/customuirw.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "zenoedit/util/log.h"
#include "variantptr.h"
#include "common.h"
#include <zenomodel/customui/customuirw.h>

using namespace zeno::iotags;
using namespace zeno::iotags::curve;


ZsgReader::ZsgReader()
{
}

ZsgReader& ZsgReader::getInstance()
{
    static ZsgReader reader;
    return reader;
}

bool ZsgReader::importNodes(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& nodeJson, const QPointF& targetPos, IAcceptor* pAcceptor)
{
    rapidjson::Document doc;
    QByteArray bytes = nodeJson.toUtf8();
    doc.Parse(bytes);

    if (!doc.IsObject() || !doc.HasMember("nodes"))
        return false;

    const rapidjson::Value& nodes = doc["nodes"];
    if (nodes.IsNull())
        return false;

    bool ret = pAcceptor->setCurrentSubGraph(pModel, subgIdx);
    if (!ret)
        return false;

    QStringList idents;
    for (const auto &node : nodes.GetObject())
    {
        const QString &nodeid = node.name.GetString();
        idents.append(nodeid);
        ret = _parseNode(nodeid, node.value, NODE_DESCS(), pAcceptor);
        if (!ret)
            return false;
    }
    return true;
}

bool ZsgReader::openFile(const QString& fn, IAcceptor* pAcceptor)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("cannot open zsg file: {} ({})", fn.toStdString(),
                       file.errorString().toStdString());
        return false;
    }

    pAcceptor->setFilePath(fn);

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    if (!doc.IsObject() || !doc.HasMember("graph"))
        return false;

    const rapidjson::Value& graph = doc["graph"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}", fn.toStdString());
        return false;
    }

    ZASSERT_EXIT(doc.HasMember("descs"), false);
    NODE_DESCS nodesDescs = _parseDescs(doc["descs"]);
    ret = pAcceptor->setLegacyDescs(graph, nodesDescs);
    if (!ret) {
        return false;
    }

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        if (!_parseSubGraph(graphName, subgraph.value, nodesDescs, pAcceptor))
            return false;
    }
    pAcceptor->EndGraphs();
    pAcceptor->switchSubGraph("main");

    if (doc.HasMember("views"))
    {
        _parseViews(doc["views"], pAcceptor);
    }

    return true;
}

bool ZsgReader::_parseSubGraph(const QString& name, const rapidjson::Value& subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor)
{
    if (!subgraph.IsObject() || !subgraph.HasMember("nodes"))
        return false;

    //todo: should consider descript info. some info of outsock without connection show in descript info.
    pAcceptor->BeginSubgraph(name);

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
        _parseNode(nodeid, node.value, descriptors, pAcceptor);
    }

    //view rect
    QRectF viewRect;
    if (subgraph.HasMember("view_rect"))
    {
        const auto& obj = subgraph["view_rect"];
        if (obj.HasMember("x") && obj.HasMember("y") && obj.HasMember("width") && obj.HasMember("height"))
            viewRect = QRectF(obj["x"].GetFloat(), obj["y"].GetFloat(), obj["width"].GetFloat(), obj["height"].GetFloat());
    } 
    else if (subgraph.HasMember("view"))
    {
        const auto& obj = subgraph["view"];
        if (obj.HasMember("scale") && obj.HasMember("trans_x") && obj.HasMember("trans_y"))
        {
            qreal scale = obj["scale"].GetFloat();
            qreal trans_x = obj["trans_x"].GetFloat();
            qreal trans_y = obj["trans_y"].GetFloat();
            qreal x = trans_x;
            qreal y = trans_y;
            qreal width = 1200. / scale;
            qreal height = 1000. / scale;
            viewRect = QRectF(x, y, width, height);
        }
    }

    pAcceptor->setViewRect(viewRect);
    pAcceptor->EndSubgraph();
    return true;
}

bool ZsgReader::_parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& legacyDescs, IAcceptor* pAcceptor)
{
    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const QString& name = nameValue.GetString();

    bool bSucceed = pAcceptor->addNode(nodeid, name, legacyDescs);
    if (!bSucceed) {
        return false;
    }

    pAcceptor->initSockets(nodeid, name, legacyDescs);

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(nodeid, name, legacyDescs, objValue["inputs"], pAcceptor);
    }
    if (objValue.HasMember("params"))
    {
        _parseParams(nodeid, name, objValue["params"], pAcceptor);
    }
    if (objValue.HasMember("outputs"))
    {
        _parseOutputs(nodeid, name, objValue["outputs"], pAcceptor);
    }
    if (objValue.HasMember("customui-panel"))
    {
        _parseCustomUI(nodeid, name, false, objValue["customui-panel"], pAcceptor);
    }
    if (objValue.HasMember("customui-node"))
    {
        _parseCustomUI(nodeid, name, true, objValue["customui-node"], pAcceptor);
    }

    if (objValue.HasMember("uipos"))
    {
        auto uipos = objValue["uipos"].GetArray();
        QPointF pos = QPointF(uipos[0].GetFloat(), uipos[1].GetFloat());
        pAcceptor->setPos(nodeid, pos);
    }
    if (objValue.HasMember("options"))
    {
        auto optionsArr = objValue["options"].GetArray();
        QStringList options;
        for (int i = 0; i < optionsArr.Size(); i++)
        {
            ZASSERT_EXIT(optionsArr[i].IsString(), false);
            const QString& optName = optionsArr[i].GetString();
            options.append(optName);
        }
        pAcceptor->setOptions(nodeid, options);
    }
    if (objValue.HasMember("dict_keys"))
    {
        _parseDictKeys(nodeid, objValue["dict_keys"], pAcceptor);
    }
    if (objValue.HasMember("socket_keys"))
    {
        _parseBySocketKeys(nodeid, objValue, pAcceptor);
    }
    if (objValue.HasMember("color_ramps"))
    {
        _parseColorRamps(nodeid, objValue["color_ramps"], pAcceptor);
    }
    if (objValue.HasMember("points") && objValue.HasMember("handlers"))
    {
        _parseLegacyCurves(nodeid, objValue["points"], objValue["handlers"], pAcceptor);
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
        blackboard.background =  QColor(blackBoardValue.HasMember("background") ? blackBoardValue["background"].GetString() : "");

        if (blackBoardValue.HasMember("width") && blackBoardValue.HasMember("height")) {
            qreal w = blackBoardValue["width"].GetFloat();
            qreal h = blackBoardValue["height"].GetFloat();
            blackboard.sz = QSizeF(w, h);
        }
        if (blackBoardValue.HasMember("params")) {
            //todo
        }
        if (blackBoardValue.HasMember("items")) {
            auto item_keys = blackBoardValue["items"].GetArray();
            for (int i = 0; i < item_keys.Size(); i++) {
                QString key = item_keys[i].GetString();
                blackboard.items.append(key);
            }
        }

        pAcceptor->setBlackboard(nodeid, blackboard);
    }

    return true;
}

void ZsgReader::_parseViews(const rapidjson::Value& jsonViews, IAcceptor* pAcceptor)
{
    if (jsonViews.HasMember("timeline"))
    {
        _parseTimeline(jsonViews["timeline"], pAcceptor);
    }
}

void ZsgReader::_parseTimeline(const rapidjson::Value& jsonTimeline, IAcceptor* pAcceptor)
{
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::start_frame) && jsonTimeline[timeline::start_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::end_frame) && jsonTimeline[timeline::end_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::curr_frame) && jsonTimeline[timeline::curr_frame].IsInt());
    ZASSERT_EXIT(jsonTimeline.HasMember(timeline::always) && jsonTimeline[timeline::always].IsBool());

    TIMELINE_INFO info;
    info.beginFrame = jsonTimeline[timeline::start_frame].GetInt();
    info.endFrame = jsonTimeline[timeline::end_frame].GetInt();
    info.currFrame = jsonTimeline[timeline::curr_frame].GetInt();
    info.bAlways = jsonTimeline[timeline::always].GetBool();

    pAcceptor->setTimeInfo(info);
}

void ZsgReader::_parseDictKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor)
{
    ZASSERT_EXIT(objValue.HasMember("inputs") && objValue["inputs"].IsArray());
    auto input_keys = objValue["inputs"].GetArray();
    for (int i = 0; i < input_keys.Size(); i++)
    {
        QString key = input_keys[i].GetString();
        pAcceptor->addDictKey(id, key, true);
    }

    ZASSERT_EXIT(objValue.HasMember("outputs") && objValue["outputs"].IsArray());
    auto output_keys = objValue["outputs"].GetArray();
    for (int i = 0; i < output_keys.Size(); i++)
    {
        QString key = output_keys[i].GetString();
        pAcceptor->addDictKey(id, key, false);
    }
}

void ZsgReader::_parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor)
{
    auto socket_keys = objValue["socket_keys"].GetArray();
    QStringList socketKeys;
    for (int i = 0; i < socket_keys.Size(); i++)
    {
        QString key = socket_keys[i].GetString();
        socketKeys.append(key);
    }
    pAcceptor->setSocketKeys(id, socketKeys);
}

void ZsgReader::_parseInputs(const QString& id, const QString& nodeName, const NODE_DESCS& legacyDescs, const rapidjson::Value& inputs, IAcceptor* pAcceptor)
{
    for (const auto& inObj : inputs.GetObject())
    {
        const QString& inSock = inObj.name.GetString();
        const auto& inputObj = inObj.value;
        if (inputObj.IsArray())
        {
            //legacy case, like [xxx-node, xxx-socket, defl]
            const auto& arr = inputObj.GetArray();
            //ZASSERT_EXIT(arr.Size() >= 2 && arr.Size() <= 3);

            QString outId, outSock;
            int n = arr.Size();
            ZASSERT_EXIT(n == 3);

            if (arr[0].IsString())
                outId = arr[0].GetString();
            if (arr[1].IsString())
                outSock = arr[1].GetString();
            pAcceptor->setInputSocket(nodeName, id, inSock, outId, outSock, arr[2]);
        }
        else if (inputObj.IsNull())
        {
            pAcceptor->setInputSocket(nodeName, id, inSock, "", "", rapidjson::Value());
        }
        else if (inputObj.IsObject())
        {
            _parseSocket(id, nodeName, inSock, true, inputObj, pAcceptor);
        }
        else
        {
            Q_ASSERT(false);
        }
    }
    pAcceptor->endInputs(id, nodeName);
}

void ZsgReader::_parseSocket(
        const QString& id,
        const QString& nodeName,
        const QString& inSock,
        bool bInput,
        const rapidjson::Value& sockObj,
        IAcceptor* pAcceptor)
{
    int sockprop = SOCKPROP_NORMAL;
    QString sockProp;
    if (sockObj.HasMember("property"))
    {
        ZASSERT_EXIT(sockObj["property"].IsString());
        sockProp = QString::fromLocal8Bit(sockObj["property"].GetString());
    }
    pAcceptor->addSocket(bInput, id, inSock, sockProp);

    QString link;
    if (sockObj.HasMember("link") && sockObj["link"].IsString())
        link = QString::fromLocal8Bit(sockObj["link"].GetString());
    if (sockObj.HasMember("default-value"))
    {
        pAcceptor->setInputSocket2(nodeName, id, inSock, link, sockProp, sockObj["default-value"]);
    }
    else
    {
        pAcceptor->setInputSocket2(nodeName, id, inSock, link, sockProp, rapidjson::Value());
    }

    if (sockObj.HasMember("dictlist-panel"))
    {
        _parseDictPanel(bInput, sockObj["dictlist-panel"], id, inSock, nodeName, pAcceptor);
    }
}

void ZsgReader::_parseDictPanel(
            bool bInput,
            const rapidjson::Value& dictPanelObj, 
            const QString& id,
            const QString& inSock,
            const QString& nodeName,
            IAcceptor* pAcceptor)
{
    if (dictPanelObj.HasMember("collasped") && dictPanelObj["collasped"].IsBool())
    {
        bool val = dictPanelObj["collasped"].GetBool();
        pAcceptor->setDictPanelProperty(bInput, id, inSock, val);
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
                link = QString::fromLocal8Bit(inputObj["link"].GetString());
            }
            pAcceptor->addInnerDictKey(true, id, inSock, keyName, link);
            if (bInput)
            {
                QString sockGrp = inSock + "/" + keyName;   //dict key io specific.
                pAcceptor->setInputSocket2(nodeName, id, sockGrp, link, "editable", rapidjson::Value());
            }
        }
    }
}

void ZsgReader::_parseParams(const QString& id, const QString& nodeName, const rapidjson::Value& jsonParams, IAcceptor* pAcceptor)
{
    if (jsonParams.IsObject())
    {
        for (const auto& paramObj : jsonParams.GetObject())
        {
            const QString& name = paramObj.name.GetString();
            const rapidjson::Value& val = paramObj.value;
            pAcceptor->setParamValue(id, nodeName, name, val);
        }
        pAcceptor->endParams(id, nodeName);
    } else {
        if (nodeName == "Blackboard" && jsonParams.IsArray())
        {
            //deprecate by zeno-old.
            return;
        }
        zeno::log_warn("not object json param");
    }
}

void ZsgReader::_parseOutputs(const QString &id, const QString &nodeName, const rapidjson::Value& outputs, IAcceptor *pAcceptor)
{
    for (const auto& outObj : outputs.GetObject())
    {
        const QString& outSock = outObj.name.GetString();
        const auto& sockObj = outObj.value;
        if (sockObj.IsObject())
        {
            if (sockObj.HasMember("dictlist-panel")) {
                _parseDictPanel(false, sockObj["dictlist-panel"], id, outSock, nodeName, pAcceptor);
            }
        }
    }
}

void ZsgReader::_parseCustomUI(const QString& id, const QString& nodeName, bool bNodeUI, const rapidjson::Value& jsonCutomUI, IAcceptor* pAcceptor)
{
    VPARAM_INFO invisibleRoot = zenomodel::importCustomUI(jsonCutomUI);
    pAcceptor->addCustomUI(id, bNodeUI, invisibleRoot);
}

void ZsgReader::_parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor)
{
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
    pAcceptor->setColorRamps(id, colorRamps);
}

void ZsgReader::_parseLegacyCurves(const QString& id,
                                   const rapidjson::Value& jsonPoints,
                                   const rapidjson::Value& jsonHandlers,
                                   IAcceptor* pAcceptor)
{
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

    pAcceptor->setLegacyCurve(id, pts, hdls);
}

void ZsgReader::_parseCurvePoints(const QString& id, const rapidjson::Value& jsonPoints, IAcceptor* pAcceptor)
{
    if (jsonPoints.IsNull())
        return;

    QVector<QPointF> pts;
    RAPIDJSON_ASSERT(jsonPoints.IsArray());
    const auto& arr = jsonPoints.GetArray();
    for (int i = 0; i < arr.Size(); i++)
    {
        const auto& pointObj = arr[i];
        bool bSucceed = false;
        QPointF pt = UiHelper::parsePoint(pointObj, bSucceed);
        ZASSERT_EXIT(bSucceed);
        pts.append(pt);
    }
}

void ZsgReader::_parseCurveHandlers(const QString& id, const rapidjson::Value& jsonHandlers, IAcceptor* pAcceptor)
{
    if (jsonHandlers.IsNull())
        return;

    RAPIDJSON_ASSERT(jsonHandlers.IsArray());
    const auto& arr = jsonHandlers.GetArray();
    QVector<QPair<QPointF, QPointF>> hdls;
    for (int i = 0; i < arr.Size(); i++)
    {
        RAPIDJSON_ASSERT(arr[i].IsArray() && arr[i].Size() == 2);
        const auto& arr_ = arr[i].GetArray();

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
    NODE_DESCS _descs;
    for (const auto& node : jsonDescs.GetObject())
    {
        const QString& name = node.name.GetString();
        const auto& objValue = node.value;

        NODE_DESC desc;
        desc.name = name;
        if (objValue.HasMember("inputs") && objValue["inputs"].IsArray())
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
                        PARAM_CONTROL ctrlType = UiHelper::getControlByType(socketType);
                        INPUT_SOCKET inputSocket;
                        inputSocket.info = SOCKET_INFO("", socketName);
                        inputSocket.info.type = socketType;
                        inputSocket.info.control = ctrlType;
                        inputSocket.info.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                        desc.inputs.insert(socketName, inputSocket);
                    }
                }
            }
        }
        if (objValue.HasMember("params") && objValue["params"].IsArray())
        {
            auto params = objValue["params"].GetArray();
            for (int i = 0; i < params.Size(); i++)
            {
                if (params[i].IsArray())
                {
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
                        PARAM_CONTROL ctrlType = UiHelper::getControlByType(socketType);
                        PARAM_INFO paramInfo;
                        paramInfo.bEnableConnect = false;
                        paramInfo.control = ctrlType;
                        paramInfo.name = socketName;
                        paramInfo.typeDesc = socketType;
                        paramInfo.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                        desc.params.insert(socketName, paramInfo);
                    }
                }
            }
        }
        if (objValue.HasMember("outputs") && objValue["outputs"].IsArray())
        {
            auto outputs = objValue["outputs"].GetArray();
            for (int i = 0; i < outputs.Size(); i++)
            {
                if (outputs[i].IsArray())
                {
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
                        PARAM_CONTROL ctrlType = UiHelper::getControlByType(socketType);
                        OUTPUT_SOCKET outputSocket;
                        outputSocket.info = SOCKET_INFO("", socketName);
                        outputSocket.info.type = socketType;
                        outputSocket.info.control = ctrlType;
                        outputSocket.info.defaultValue = UiHelper::parseStringByType(socketDefl, socketType);
                        desc.outputs.insert(socketName, outputSocket);
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

        _descs.insert(name, desc);
    }
    return _descs;
}


