#include "zsgreader.h"
#include "../../zenoui/util/uihelper.h"
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "zenoedit/util/log.h"
#include <zenoui/model/variantptr.h>
#include <zenoui/model/curvemodel.h>

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

bool ZsgReader::openFile(const QString& fn, IAcceptor* pAcceptor)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        zeno::log_error("canot open zsg file: {} ({})", fn.toStdString(),
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
    pAcceptor->setLegacyDescs(graph, nodesDescs);

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        if (!_parseSubGraph(graphName, subgraph.value, nodesDescs, pAcceptor))
            return false;
    }
    pAcceptor->switchSubGraph("main");
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

void ZsgReader::_parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& legacyDescs, IAcceptor* pAcceptor)
{
    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const QString& name = nameValue.GetString();

    bool bSucceed = pAcceptor->addNode(nodeid, name, legacyDescs);
    if (!bSucceed) {
        return;
    }

    pAcceptor->initSockets(nodeid, name, legacyDescs);

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(nodeid, name, legacyDescs, objValue["inputs"], pAcceptor);
    }

    if (objValue.HasMember("params"))
    {
        _parseParams(nodeid, objValue["params"], pAcceptor);
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
            ZASSERT_EXIT(optionsArr[i].IsString());
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

        pAcceptor->setBlackboard(nodeid, blackboard);
    }
}

QVariant ZsgReader::_parseDefaultValue(const QString& defaultValue, const QString& type)
{
    return UiHelper::_parseDefaultValue(defaultValue, type);
    //some data like vec3f, cast to string first.
    //bool bOk = false;
    //double val = defaultValue.toDouble(&bOk);
    //QVariant var;
    //if (bOk) {
        //var = val;
    //} else {
        //var = defaultValue;
    //}
    //return var;
}

QVariant ZsgReader::_parseToVariant(const QString& type, const rapidjson::Value& val, QObject* parentRef)
{
    if (val.GetType() == rapidjson::kStringType)
    {
		return val.GetString();
    }
	else if (val.GetType() == rapidjson::kNumberType)
    {
        //if (val.IsInt())
            //zeno::log_critical("happy {}", val.GetInt());
        if (val.IsDouble())
            return val.GetDouble();
        else if (val.IsInt())
            return val.GetInt();
        else {
            zeno::log_warn("bad rapidjson number type {}", val.GetType());
            return QVariant();
        }
	}
	else if (val.GetType() == rapidjson::kTrueType)
    {
		return val.GetBool();
	}
	else if (val.GetType() == rapidjson::kFalseType)
    {
		return val.GetBool();
	}
	else if (val.GetType() == rapidjson::kNullType)
    {
		return QVariant();
    }
    else if (val.GetType() == rapidjson::kArrayType)
    {
        QVector<qreal> vec;
        auto values = val.GetArray();
        for (int i = 0; i < values.Size(); i++)
        {
            vec.append(values[i].GetFloat());
        }
        return QVariant::fromValue(vec);
    }
    else if (val.GetType() == rapidjson::kObjectType)
    {
	    if (type == "curve")
        {
            CurveModel *pModel = _parseCurveModel(val, parentRef);
            return QVariantPtr<CurveModel>::asVariant(pModel);
        }
    }

    zeno::log_warn("bad rapidjson value type {}", val.GetType());
    return QVariant();
}

CurveModel* ZsgReader::_parseCurveModel(const rapidjson::Value& jsonCurve, QObject* parentRef)
{
    ZASSERT_EXIT(jsonCurve.HasMember(key_objectType), nullptr);
    QString type = jsonCurve[key_objectType].GetString();
    if (type != "curve") {
        return nullptr;
    }

    ZASSERT_EXIT(jsonCurve.HasMember(key_range), nullptr);
    const rapidjson::Value &rgObj = jsonCurve[key_range];
    ZASSERT_EXIT(rgObj.HasMember(key_xFrom) && rgObj.HasMember(key_xTo) && rgObj.HasMember(key_yFrom) && rgObj.HasMember(key_yTo), nullptr);

    CURVE_RANGE rg;
    ZASSERT_EXIT(rgObj[key_xFrom].IsDouble() && rgObj[key_xTo].IsDouble() && rgObj[key_yFrom].IsDouble() && rgObj[key_yTo].IsDouble(), nullptr);
    rg.xFrom = rgObj[key_xFrom].GetDouble();
    rg.xTo = rgObj[key_xTo].GetDouble();
    rg.yFrom = rgObj[key_yFrom].GetDouble();
    rg.yTo = rgObj[key_yTo].GetDouble();

    //todo: id
    CurveModel* pModel = new CurveModel("x", rg, parentRef); 

    if (jsonCurve.HasMember(key_timeline) && jsonCurve[key_timeline].IsBool())
    {
        bool bTimeline = jsonCurve[key_timeline].GetBool();
        pModel->setTimeline(bTimeline);
    }

    ZASSERT_EXIT(jsonCurve.HasMember(key_nodes), nullptr);
    for (const rapidjson::Value &nodeObj : jsonCurve[key_nodes].GetArray())
    {
        ZASSERT_EXIT(nodeObj.HasMember("x") && nodeObj["x"].IsDouble(), nullptr);
        ZASSERT_EXIT(nodeObj.HasMember("y") && nodeObj["y"].IsDouble(), nullptr);
        QPointF pos(nodeObj["x"].GetDouble(), nodeObj["y"].GetDouble());

        ZASSERT_EXIT(nodeObj.HasMember(key_left_handle) && nodeObj[key_left_handle].IsObject(), nullptr);
        auto leftHdlObj = nodeObj[key_left_handle].GetObject();
        ZASSERT_EXIT(leftHdlObj.HasMember("x") && leftHdlObj.HasMember("y"), nullptr);
        qreal leftX = leftHdlObj["x"].GetDouble();
        qreal leftY = leftHdlObj["y"].GetDouble();
        QPointF leftOffset(leftX, leftY);

        ZASSERT_EXIT(nodeObj.HasMember(key_right_handle) && nodeObj[key_right_handle].IsObject(), nullptr);
        auto rightHdlObj = nodeObj[key_right_handle].GetObject();
        ZASSERT_EXIT(rightHdlObj.HasMember("x") && rightHdlObj.HasMember("y"), nullptr);
        qreal rightX = rightHdlObj["x"].GetDouble();
        qreal rightY = rightHdlObj["y"].GetDouble();
        QPointF rightOffset(rightX, rightY);

        HANDLE_TYPE hdlType = HDL_ASYM;
        if (nodeObj.HasMember(key_type) && nodeObj[key_type].IsString())
        {
            QString type = nodeObj[key_type].GetString();
            if (type == "aligned") {
                hdlType = HDL_ALIGNED;
            } else if (type == "asym") {
                hdlType = HDL_ASYM;
            } else if (type == "free") {
                hdlType = HDL_FREE;
            } else if (type == "vector") {
                hdlType = HDL_VECTOR;
            }
        }

        bool bLockX = (nodeObj.HasMember(key_lockX) && nodeObj[key_lockX].IsBool());
        bool bLockY = (nodeObj.HasMember(key_lockY) && nodeObj[key_lockY].IsBool());

        QStandardItem *pItem = new QStandardItem;
        pItem->setData(pos, ROLE_NODEPOS);
        pItem->setData(leftOffset, ROLE_LEFTPOS);
        pItem->setData(rightOffset, ROLE_RIGHTPOS);
        pItem->setData(hdlType, ROLE_TYPE);
        pModel->appendRow(pItem);
    }
    return pModel;
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
            const auto& arr = inputObj.GetArray();
            //ZASSERT_EXIT(arr.Size() >= 2 && arr.Size() <= 3);

            QString outId, outSock;
            QVariant defaultValue;
            if (arr.Size() > 0 && arr[0].IsString())
                outId = arr[0].GetString();
            if (arr.Size() > 1 && arr[1].IsString())
                outSock = arr[1].GetString();
            if (arr.Size() > 2)
                pAcceptor->setInputSocket(nodeName, id, inSock, outId, outSock, arr[2], legacyDescs);
        }
        else if (inputObj.IsNull())
        {
            pAcceptor->setInputSocket(nodeName, id, inSock, "", "", rapidjson::Value(), legacyDescs);
        }
        else
        {
            Q_ASSERT(false);
        }
    }
}

void ZsgReader::_parseParams(const QString& id, const rapidjson::Value& jsonParams, IAcceptor* pAcceptor)
{
    if (jsonParams.IsObject())
    {
        for (const auto& paramObj : jsonParams.GetObject())
        {
            const QString& name = paramObj.name.GetString();
            const rapidjson::Value& val = paramObj.value;
            QVariant var = _parseToVariant("", val, pAcceptor->currGraphObj()); //todo: fill type.
            pAcceptor->setParamValue(id, name, var);
        }
    } else {
        zeno::log_warn("not object json param");
    }
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
                        PARAM_CONTROL ctrlType = UiHelper::_getControlType(socketType);
                        INPUT_SOCKET inputSocket;
                        inputSocket.info = SOCKET_INFO("", socketName);
                        inputSocket.info.type = socketType;
                        inputSocket.info.control = ctrlType;
                        inputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);
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
                        PARAM_CONTROL ctrlType = UiHelper::_getControlType(socketType);
                        PARAM_INFO paramInfo;
                        paramInfo.bEnableConnect = false;
                        paramInfo.control = ctrlType;
                        paramInfo.name = socketName;
                        paramInfo.typeDesc = socketType;
                        paramInfo.defaultValue = _parseDefaultValue(socketDefl, socketType);
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
                        PARAM_CONTROL ctrlType = UiHelper::_getControlType(socketType);
                        OUTPUT_SOCKET outputSocket;
                        outputSocket.info = SOCKET_INFO("", socketName);
                        outputSocket.info.type = socketType;
                        outputSocket.info.control = ctrlType;
                        outputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);
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
