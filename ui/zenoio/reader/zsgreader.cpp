#include "zsgreader.h"
#include "../../zenoui/util/uihelper.h"
#include <zeno/utils/logger.h>
#include "zenoedit/util/log.h"
#include <zenoui/model/variantptr.h>
#include <zenoui/model/curvemodel.h>


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

    const rapidjson::Value& graph = doc["graph"];
    if (graph.IsNull()) {
        zeno::log_error("json format incorrect in zsg file: {}", fn.toStdString());
        return false;
    }

    NODE_DESCS nodesDescs = _parseDescs(doc["descs"]);
    pAcceptor->setDescriptors(nodesDescs);

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        _parseSubGraph(graphName, subgraph.value, nodesDescs, pAcceptor);
    }
    pAcceptor->switchSubGraph("main");
    return true;
}

void ZsgReader::_parseSubGraph(const QString& name, const rapidjson::Value& subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor)
{
    //todo: should consider descript info. some info of outsock without connection show in descript info.
    pAcceptor->BeginSubgraph(name);

    const auto& nodes = subgraph["nodes"];
    if (nodes.IsNull())
        return;

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
        ZASSERT_EXIT(obj.HasMember("scale") && obj.HasMember("trans_x") && obj.HasMember("trans_y"));
        qreal scale = obj["scale"].GetFloat();
        qreal trans_x = obj["trans_x"].GetFloat();
        qreal trans_y = obj["trans_y"].GetFloat();
        qreal x = trans_x;
        qreal y = trans_y;
        qreal width = 1200. / scale;
        qreal height = 1000. / scale;
        viewRect = QRectF(x, y, width, height);
    }
    pAcceptor->setViewRect(viewRect);

    pAcceptor->EndSubgraph();
}

void ZsgReader::_parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor)
{
    const auto& objValue = nodeObj;
    const rapidjson::Value& nameValue = objValue["name"];
    const QString& name = nameValue.GetString();

    if (descriptors.find(name) == descriptors.end())
    {
        qDebug() << QString("no node class named [%1]").arg(name);
        return;
    }

    pAcceptor->addNode(nodeid, nameValue.GetString(), descriptors);

    //socket_keys should be inited before socket init.
    if (objValue.HasMember("socket_keys"))
    {
        _parseBySocketKeys(nodeid, objValue, pAcceptor);
    }
    pAcceptor->initSockets(nodeid, name, descriptors);

    if (objValue.HasMember("inputs"))
    {
        _parseInputs(nodeid, name, descriptors, objValue["inputs"], pAcceptor);
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
    if (objValue.HasMember("socket_keys"))
    {
        auto socket_keys = objValue["socket_keys"].GetArray();
        QStringList socketKeys;
        for (int i = 0; i < socket_keys.Size(); i++)
        {
            socketKeys.append(socket_keys[i].GetString());
        }
        pAcceptor->setSocketKeys(nodeid, socketKeys);
    }
    if (objValue.HasMember("color_ramps"))
    {
        _parseColorRamps(nodeid, objValue["color_ramps"], pAcceptor);
    }
    if (name == "Blackboard")
    {
        BLACKBOARD_INFO blackboard;
        QString title, content;
        QSizeF sz;
        bool special = false;

        if (objValue.HasMember("special"))
        {
            blackboard.special = objValue["special"].GetBool();
        }

        title = objValue.HasMember("title") ? objValue["title"].GetString() : "";
        content = objValue.HasMember("content") ? objValue["content"].GetString() : "";

        if (objValue.HasMember("width") && objValue.HasMember("height"))
        {
            qreal w = objValue["width"].GetFloat();
            qreal h = objValue["height"].GetFloat();
            blackboard.sz = QSizeF(w, h);
        }
        if (objValue.HasMember("params"))
        {
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
    ZASSERT_EXIT(jsonCurve.HasMember("range"), nullptr);
    const rapidjson::Value& rgObj = jsonCurve["range"];
    ZASSERT_EXIT(rgObj.HasMember("xFrom") && rgObj.HasMember("xTo") && rgObj.HasMember("yFrom") && rgObj.HasMember("yTo"), nullptr);

    CURVE_RANGE rg;
    ZASSERT_EXIT(rgObj["xFrom"].IsDouble() && rgObj["xTo"].IsDouble() && rgObj["yFrom"].IsDouble() && rgObj["yTo"].IsDouble(), nullptr);
    rg.xFrom = rgObj["xFrom"].GetDouble();
    rg.xTo = rgObj["xTo"].GetDouble();
    rg.yFrom = rgObj["yFrom"].GetDouble();
    rg.yTo = rgObj["yTo"].GetDouble();

    //todo: id
    CurveModel* pModel = new CurveModel("x", rg, parentRef); 

    if (jsonCurve.HasMember("timeline") && jsonCurve["timeline"].IsBool())
    {
        bool bTimeline = jsonCurve["timeline"].GetBool();
        pModel->setTimeline(bTimeline);
    }

    ZASSERT_EXIT(jsonCurve.HasMember("nodes"), nullptr);
    for (const rapidjson::Value &nodeObj : jsonCurve["nodes"].GetArray())
    {
        ZASSERT_EXIT(nodeObj.HasMember("x") && nodeObj["x"].IsDouble(), nullptr);
        ZASSERT_EXIT(nodeObj.HasMember("y") && nodeObj["y"].IsDouble(), nullptr);
        QPointF pos(nodeObj["x"].GetDouble(), nodeObj["y"].GetDouble());

        ZASSERT_EXIT(nodeObj.HasMember("left-handle") && nodeObj["left-handle"].IsObject(), nullptr);
        auto leftHdlObj = nodeObj["left-handle"].GetObject();
        ZASSERT_EXIT(leftHdlObj.HasMember("x") && leftHdlObj.HasMember("y"), nullptr);
        qreal leftX = leftHdlObj["x"].GetDouble();
        qreal leftY = leftHdlObj["y"].GetDouble();
        QPointF leftOffset(leftX, leftY);

        ZASSERT_EXIT(nodeObj.HasMember("right-handle") && nodeObj["right-handle"].IsObject(), nullptr);
        auto rightHdlObj = nodeObj["right-handle"].GetObject();
        ZASSERT_EXIT(rightHdlObj.HasMember("x") && rightHdlObj.HasMember("y"), nullptr);
        qreal rightX = rightHdlObj["x"].GetDouble();
        qreal rightY = rightHdlObj["y"].GetDouble();
        QPointF rightOffset(rightX, rightY);

        HANDLE_TYPE hdlType = HDL_ASYM;
        if (nodeObj.HasMember("type") && nodeObj["type"].IsString())
        {
            QString type = nodeObj["type"].GetString();
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

        bool bLockX = (nodeObj.HasMember("lockX") && nodeObj["lockX"].IsBool());
        bool bLockY = (nodeObj.HasMember("lockY") && nodeObj["lockY"].IsBool());

        QStandardItem *pItem = new QStandardItem;
        pItem->setData(pos, ROLE_NODEPOS);
        pItem->setData(leftOffset, ROLE_LEFTPOS);
        pItem->setData(rightOffset, ROLE_RIGHTPOS);
        pItem->setData(hdlType, ROLE_TYPE);
        pModel->appendRow(pItem);
    }
    return pModel;
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

void ZsgReader::_parseInputs(const QString& id, const QString& nodeName, const NODE_DESCS& descriptors, const rapidjson::Value& inputs, IAcceptor* pAcceptor)
{
    ZASSERT_EXIT(descriptors.find(nodeName) != descriptors.end());
    const NODE_DESC &desc = descriptors[nodeName];
    for (const auto& inObj : inputs.GetObject())
    {
        const QString& inSock = inObj.name.GetString();
        const auto& inputObj = inObj.value;
        if (inputObj.IsArray())
        {
            const auto& arr = inputObj.GetArray();
            //ZASSERT_EXIT(arr.Size() >= 2 && arr.Size() <= 3);

            SOCKET_INFO descInfo;
            if (desc.inputs.find(inSock) != desc.inputs.end())
            {
                descInfo = desc.inputs[inSock].info;
            }

            QString outId, outSock;
            QVariant defaultValue;
            if (arr.Size() > 0 && arr[0].IsString())
                outId = arr[0].GetString();
            if (arr.Size() > 1 && arr[1].IsString())
                outSock = arr[1].GetString();
            if (arr.Size() > 2)
                defaultValue = _parseToVariant(descInfo.type, arr[2], pAcceptor->currGraphObj());
            
            pAcceptor->setInputSocket(id, inSock, outId, outSock, defaultValue);
        }
        else if (inputObj.IsNull())
        {
            pAcceptor->setInputSocket(id, inSock, "", "", QVariant());
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
        if (name == "MakeHeatmap") {
            int j;
            j = 0;
        }
        const auto& objValue = node.value;
        auto inputs = objValue["inputs"].GetArray();
        auto outputs = objValue["outputs"].GetArray();
        auto params = objValue["params"].GetArray();
        auto categories = objValue["categories"].GetArray();

        NODE_DESC desc;

        for (int i = 0; i < inputs.Size(); i++) {
            if (inputs[i].IsArray()) {
                auto input_triple = inputs[i].GetArray();
                const QString& socketType = input_triple[0].GetString();
                const QString& socketName = input_triple[1].GetString();
                //zeno::log_info("input_triple[2] = {}", input_triple[2].GetType());
                const QString& socketDefl = input_triple[2].GetString();
                PARAM_CONTROL ctrlType = UiHelper::_getControlType(socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName);
                inputSocket.info.type = socketType;
                inputSocket.info.control = ctrlType;
                inputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);
                desc.inputs.insert(socketName, inputSocket);
            }
            else {
            }
        }

        for (int i = 0; i < params.Size(); i++) {
            if (params[i].IsArray()) {
                auto param_triple = params[i].GetArray();
                const QString& socketType = param_triple[0].GetString();
                const QString& socketName = param_triple[1].GetString();
                //zeno::log_info("param_triple[2] = {}", param_triple[2].GetType());
                const QString& socketDefl = param_triple[2].GetString();
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

        for (int i = 0; i < outputs.Size(); i++) {
            if (outputs[i].IsArray()) {
                auto output_triple = outputs[i].GetArray();
                const QString& socketType = output_triple[0].GetString();
                const QString& socketName = output_triple[1].GetString();
                const QString& socketDefl = output_triple[2].GetString();
                PARAM_CONTROL ctrlType = UiHelper::_getControlType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName);
                outputSocket.info.type = socketType;
                outputSocket.info.control = ctrlType;
                outputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);

                desc.outputs.insert(socketName, outputSocket);
            }
            else {
            }
        }

        for (int i = 0; i < categories.Size(); i++)
        {
            desc.categories.push_back(categories[i].GetString());
        }

        _descs.insert(name, desc);
    }
    return _descs;
}
