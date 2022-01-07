#include "../model/nodesmodel.h"
#include "../model/modelrole.h"
#include "zsgreader.h"
#include "../util/uihelper.h"


ZsgReader::ZsgReader()
{
}

ZsgReader& ZsgReader::getInstance()
{
    static ZsgReader reader;
    return reader;
}

GraphsModel* ZsgReader::loadZsgFile(const QString& fn)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret)
        return nullptr;

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    rapidjson::Value& graph = doc["graph"];
    if (graph.IsNull())
        return nullptr;

    GraphsModel* pModel = new GraphsModel;
    pModel->setFilePath(fn);

    NODE_DESCS nodesDescs = UiHelper::parseDescs(doc["descs"]);
    pModel->setDescriptors(nodesDescs);

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        SubGraphModel *subGraphModel = _parseSubGraph(pModel, subgraph.value);
        subGraphModel->setName(graphName);
        pModel->appendSubGraph(subGraphModel);
    }
    pModel->switchSubGraph("main");

    return pModel;
}

void ZsgReader::_parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph)
{

}

SubGraphModel* ZsgReader::_parseSubGraph(GraphsModel* pGraphsModel, const rapidjson::Value &subgraph)
{
    //todo: should consider descript info. some info of outsock without connection show in descript info.
    SubGraphModel *pModel = new SubGraphModel(pGraphsModel);
    const auto& nodes = subgraph["nodes"];
    const NODE_DESCS& descriptors = pModel->descriptors();
    if (nodes.IsNull())
        return nullptr;

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
        NODE_DATA nodeData;

        const QString& nodeid = node.name.GetString();
        nodeData[ROLE_OBJID] = nodeid;

        const auto &objValue = node.value;
        const rapidjson::Value& nameValue = objValue["name"];
        const QString &name = nameValue.GetString();

        if (descriptors.find(name) == descriptors.end())
        {
            qDebug() << QString("no node class named [%1]").arg(name);
            continue;
        }

        nodeData[ROLE_OBJNAME] = nameValue.GetString();
        nodeData[ROLE_OBJTYPE] = NORMAL_NODE;
        nodeData[ROLE_COLLASPED] = false;

        if (objValue.HasMember("inputs"))
        {
            INPUT_SOCKETS inputs = descriptors[name].inputs;
            _parseInputs(inputs, descriptors, objIdToName, objValue["inputs"]);
            nodeData[ROLE_INPUTS] = QVariant::fromValue(inputs);
        }

        OUTPUT_SOCKETS outputs = descriptors[name].outputs;
        nodeData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

        if (objValue.HasMember("params"))
        {
            PARAMS_INFO params = descriptors[name].params;
            _parseParams(params, objValue["params"]);
            nodeData[ROLE_PARAMETERS] = QVariant::fromValue(params);
        }
        if (objValue.HasMember("uipos"))
        {
            auto uipos = objValue["uipos"].GetArray();
            nodeData[ROLE_OBJPOS] = QPointF(uipos[0].GetFloat(), uipos[1].GetFloat());
        }
        if (objValue.HasMember("options"))
        {
            auto optionsArr = objValue["options"].GetArray();
            for (int i = 0; i < optionsArr.Size(); i++)
            {
                Q_ASSERT(optionsArr[i].IsString());
                const QString& optName = optionsArr[i].GetString();
                int opts = 0;
                if (optName == "ONCE")
                {
                    opts |= OPT_ONCE;
                }
                else if (optName == "PREP")
                {
                    opts |= OPT_PREP;
                }
                else if (optName == "VIEW")
                {
                    opts |= OPT_VIEW;
                }
                else if (optName == "MUTE")
                {
                    opts |= OPT_MUTE;
                }
                else if (optName == "collapsed")
                {
                    nodeData[ROLE_COLLASPED] = true;
                }
                else
                {
                    Q_ASSERT(false);
                }
                nodeData[ROLE_OPTIONS] = opts;
            }
        }
        if (objValue.HasMember("color_ramps"))
        {
            COLOR_RAMPS colorRamps;
            _parseColorRamps(colorRamps, objValue["color_ramps"]);
            nodeData[ROLE_OBJTYPE] = HEATMAP_NODE;
            nodeData[ROLE_COLORRAMPS] = QVariant::fromValue(colorRamps);
        }
        if (name == "Blackboard")
        {
            nodeData[ROLE_OBJTYPE] = BLACKBOARD_NODE;
            if (objValue.HasMember("special"))
            {
                nodeData[ROLE_BLACKBOARD_SPECIAL] = objValue["special"].GetBool();
            }

            nodeData[ROLE_BLACKBOARD_TITLE] = objValue.HasMember("title") ? objValue["title"].GetString() : "";
            nodeData[ROLE_BLACKBOARD_CONTENT] = objValue.HasMember("content") ? objValue["content"].GetString() : "";

            if (objValue.HasMember("width") && objValue.HasMember("height"))
            {
                qreal w = objValue["width"].GetFloat();
                qreal h = objValue["height"].GetFloat();
                nodeData[ROLE_BLACKBOARD_SIZE] = QSizeF(w, h);
            }
            if (objValue.HasMember("params"))
            {
                //todo
            }
        }
        pModel->appendItem(nodeData);
    }
    _parseOutputConnections(pModel);

    //view rect
    QRectF viewRect;
    if (subgraph.HasMember("view_rect"))
    {
        const auto& obj = subgraph["view_rect"];
        Q_ASSERT(obj.HasMember("x") && obj.HasMember("y") && obj.HasMember("width") && obj.HasMember("height"));
        viewRect = QRectF(obj["x"].GetFloat(), obj["y"].GetFloat(), obj["width"].GetFloat(), obj["height"].GetFloat());
    } 
    else if (subgraph.HasMember("view"))
    {
        const auto& obj = subgraph["view"];
        Q_ASSERT(obj.HasMember("scale") && obj.HasMember("trans_x") && obj.HasMember("trans_y"));
        qreal scale = obj["scale"].GetFloat();
        qreal trans_x = obj["trans_x"].GetFloat();
        qreal trans_y = obj["trans_y"].GetFloat();
        qreal x = trans_x;
        qreal y = trans_y;
        qreal width = 1200. / scale;
        qreal height = 1000. / scale;
        viewRect = QRectF(x, y, width, height);
    }
    pModel->setViewRect(viewRect);

    return pModel;
}

QString ZsgReader::dumpNodeData(const NODE_DATA& data)
{
    //not compatible with current zeno file format, only used for copy/paste.
    //but now copy/paste is limited in current process.

    QString result;

    QJsonObject obj;

    //ROLE_OBJID
    QString nodeid = data[ROLE_OBJID].toString();
    obj.insert("id", nodeid);

    //ROLE_OBJNAME
    QString name = data[ROLE_OBJNAME].toString();
    obj.insert("name", name);

    //ROLE_INPUTS
    const INPUT_SOCKETS& inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    QJsonObject inputsArr;
    for (auto inputSocket : inputs)
    {
        const QString& inSock = inputSocket.info.name;
        QJsonArray arr;
        arr.push_back(QJsonValue::Null);
        arr.push_back(QJsonValue::Null);
        arr.push_back(QJsonValue::Null);
        inputsArr.insert(inSock, arr);
    }
    obj.insert("inputs", inputsArr);

    //ROLE_OUTPUTS
    const OUTPUT_SOCKETS &outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    QJsonObject outputsArr;
    for (auto ouputSocket : outputs)
    {
        const QString& outSock = ouputSocket.info.name;
        QJsonArray arr;
        arr.push_back(QJsonValue::Null);
        arr.push_back(QJsonValue::Null);
        arr.push_back(QJsonValue::Null);
        outputsArr.insert(outSock, arr);
    }
    obj.insert("outputs", outputsArr);

    //ROLE_PARAMETERS
    const PARAMS_INFO &params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
    QJsonObject paramsObj;
    for (PARAM_INFO paramInfo : params)
    {
        //todo: different type.
        paramsObj.insert(paramInfo.name, paramInfo.value.toString());
    }
    obj.insert("params", paramsObj);

    //ROLE_OBJPOS
    QPointF pos = data[ROLE_OBJPOS].toPointF();
    QJsonArray posArr;
    posArr.push_back(pos.x());
    posArr.push_back(pos.y());
    obj.insert("uipos", posArr);

    //ROLE_OBJTYPE
    NODE_TYPE type = (NODE_TYPE)data[ROLE_OBJTYPE].toInt();
    obj.insert("type", type);

    //ROLE_COLORRAMPS

    //ROLE_BLACKBOARD_SPECIAL

    //ROLE_BLACKBOARD_TITLE

    //ROLE_BLACKBOARD_CONTENT

    //ROLE_BLACKBOARD_SIZE

    QJsonDocument doc(obj);
    QString strJson(doc.toJson(QJsonDocument::Compact));
    return strJson;
}

NODE_DATA ZsgReader::importNodeData(const QString json)
{
    NODE_DATA data;
    QJsonObject obj;
    QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    if (!doc.isNull())
    {
        QString nodeid = doc["id"].toString();
        data[ROLE_OBJID] = nodeid;
        data[ROLE_OBJNAME] = doc["name"].toString();
        data[ROLE_OBJTYPE] = doc["type"].toInt();
        QJsonArray arr = doc["uipos"].toArray();
        data[ROLE_OBJPOS] = QPointF(arr[0].toDouble(), arr[1].toDouble());
        
        QJsonObject inputs = doc["inputs"].toObject();
        INPUT_SOCKETS inputSockets;
        for (auto key : inputs.keys())
        {
            INPUT_SOCKET socket;
            socket.info.name = key;
            socket.info.nodeid = nodeid;
            inputSockets[key] = socket;
        }
        data[ROLE_INPUTS] = QVariant::fromValue(inputSockets);

        OUTPUT_SOCKETS outputSockets;
        QJsonObject outputs = doc["outputs"].toObject();
        for (auto key : outputs.keys())
        {
            OUTPUT_SOCKET socket;
            socket.info.name = key;
            socket.info.nodeid = nodeid;
            outputSockets[key] = socket;
        }
        data[ROLE_OUTPUTS] = QVariant::fromValue(outputSockets);

        PARAMS_INFO paramsInfo;
        QJsonObject params = doc["params"].toObject();
        for (auto key : params.keys())
        {
            PARAM_INFO param;
            param.name = key;
            param.value = params[key].toString();
            paramsInfo.insert(key, param);
        }
        data[ROLE_PARAMETERS] = QVariant::fromValue(paramsInfo);
    }
    return data;
}

PARAM_CONTROL ZsgReader::_getControlType(const QString& type)
{
    if (type == "int") {
        return CONTROL_INT;
    } else if (type == "bool") {
        return CONTROL_BOOL;
    } else if (type == "float") {
        return CONTROL_FLOAT;
    } else if (type == "string") {
        return CONTROL_STRING;
    } else if (type == "writepath") {
        return CONTROL_WRITEPATH;
    } else if (type == "readpath") {
        return CONTROL_READPATH;
    } else if (type == "multiline_string") {
        return CONTROL_MULTILINE_STRING;
    } else if (type == "_RAMPS") {
        return CONTROL_HEAPMAP;
    } else if (type.startsWith("enum ")) {
        return CONTROL_ENUM;
    } else {
        return CONTROL_NONE;
    }
}

QVariant ZsgReader::_parseDefaultValue(const QString& defaultValue)
{
    //some data like vec3f, cast to string first.
    bool bOk = false;
    float val = defaultValue.toFloat(&bOk);
    QVariant var;
    if (bOk) {
        var = val;
    } else {
        var = defaultValue;
    }
    return var;
}

void ZsgReader::_parseOutputConnections(SubGraphModel* pModel)
{
    //init output ports for each node.
    int n = pModel->rowCount();
    for (int r = 0; r < n; r++)
    {
        const QModelIndex &idx = pModel->index(r, 0);
        const QString &inNode = idx.data(ROLE_OBJID).toString();
        INPUT_SOCKETS inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        foreach (const QString& inSockName, inputs.keys())
        {
            const INPUT_SOCKET& inSocket = inputs[inSockName];
            for (const QString& outNode : inSocket.outNodes.keys())
            {
                for (const QString& outSock : inSocket.outNodes[outNode].keys())
                {
                    const QModelIndex &outIdx = pModel->index(outNode);
                    OUTPUT_SOCKETS outputs = pModel->data(outIdx, ROLE_OUTPUTS).value<OUTPUT_SOCKETS>(); 
                    outputs[outSock].inNodes[inNode][inSockName] = SOCKET_INFO(inNode, inSockName);
                    pModel->setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
                }
            }
        }
    }
}

void ZsgReader::_parseInputs(INPUT_SOCKETS& inputSockets, const NODE_DESCS& descriptors,
    const QMap<QString, QString>& objId2Name, const rapidjson::Value &inputs)
{
    const auto &inputsObj = inputs.GetObject();
    for (INPUT_SOCKET& inputSocket : inputSockets)
    {
        QByteArray bytes = inputSocket.info.name.toUtf8();
        const auto &inputObj = inputsObj[bytes.data()];
        if (inputObj.IsArray())
        {
            const auto &arr = inputsObj[bytes.data()].GetArray();
            RAPIDJSON_ASSERT(arr.Size() >= 2);
            if (!arr[0].IsNull())
            {
                const QString& outId = arr[0].GetString();
                //outNode may be lose descriptor and not built and stored in model.
                const QString& nodeName = objId2Name[outId];
                if (!nodeName.isEmpty() && descriptors.find(nodeName) == descriptors.end())
                {
                    continue;
                }

                if (!arr[1].IsNull()) {
                    const QString socketName = arr[1].GetString();
                    inputSocket.outNodes[outId][socketName] = SOCKET_INFO(outId, socketName);
                }
                //to ask: default value type else
                if (arr.Size() > 2) {
                    if (arr[2].GetType() == rapidjson::kStringType) {
                        inputSocket.info.defaultValue = arr[2].GetString();
                    } else if (arr[2].GetType() == rapidjson::kNumberType) {
                        inputSocket.info.defaultValue = arr[2].GetFloat();
                    }
                    else if (arr[2].GetType() == rapidjson::kTrueType) {
                        inputSocket.info.defaultValue = arr[2].GetBool();
                    }
                    else if (arr[2].GetType() == rapidjson::kFalseType) {
                        inputSocket.info.defaultValue = arr[2].GetBool();
                    }
                }
            }
            else
            {
                if (arr.Size() > 2) {
					if (arr[2].GetType() == rapidjson::kStringType) {
						inputSocket.info.defaultValue = arr[2].GetString();
					}
					else if (arr[2].GetType() == rapidjson::kNumberType) {
						inputSocket.info.defaultValue = arr[2].GetFloat();
                        QVariant::Type varType = inputSocket.info.defaultValue.type();
                        if (varType == QMetaType::Float)
                        {
                            int j;
                            j = 0;
                        }
					}
					else if (arr[2].GetType() == rapidjson::kTrueType) {
						inputSocket.info.defaultValue = arr[2].GetBool();
					}
					else if (arr[2].GetType() == rapidjson::kFalseType) {
						inputSocket.info.defaultValue = arr[2].GetBool();
					}
                }
            }
        }
    }
}

void ZsgReader::_parseParams(PARAMS_INFO& params, const rapidjson::Value& jsonParams)
{
    if (jsonParams.IsObject())
    {
        const auto &paramsObj = jsonParams.GetObject();
        for (PARAM_INFO &param : params) {
            const QString &name = param.name;
            QByteArray bytes = name.toUtf8();
            const auto &paramObj = paramsObj[bytes.data()];
            rapidjson::Type type = paramObj.GetType();
            param.bEnableConnect = false;
            if (type == rapidjson::kNullType) {
            } else if (type == rapidjson::kStringType) {
                param.value = paramObj.GetString();
            } else if (type == rapidjson::kNumberType) {
                param.value = paramObj.GetDouble();
			}
			else if (type == rapidjson::kTrueType) {
                param.value = paramObj.GetBool();
			}
			else if (type == rapidjson::kFalseType) {
                param.value = paramObj.GetBool();
			}
        }
    }
    

    //TODO: input data may be part of param in the future and vice versa, 
    /*
    for (auto inSock : inputs)
    {
        PARAM_INFO param;
        param.defaultValue = inSock.defaultValue;
        param.bEnableConnect = true;
        param.name = inSock.info.name;
        param.value = QVariant();   //current set to null when no calc started.
        params.insert(param.name, param);
    }
    */
}

void ZsgReader::_parseColorRamps(COLOR_RAMPS& colorRamps, const rapidjson::Value& jsonColorRamps)
{
    if (jsonColorRamps.IsNull())
        return;

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