#include "zsgreader.h"
#include <model/modelrole.h>


ZsgReader::ZsgReader()
{
}

ZsgReader& ZsgReader::getInstance()
{
    static ZsgReader reader;
    return reader;
}

void ZsgReader::loadZsgFile(const QString& fn, IAcceptor* pAcceptor)
{
    QFile file(fn);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret)
        return;

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    rapidjson::Value& graph = doc["graph"];
    if (graph.IsNull())
        return;

    NODE_DESCS nodesDescs = _parseDescs(doc["descs"]);
    pAcceptor->setDescriptors(nodesDescs);

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        _parseSubGraph(graphName, subgraph, nodesDescs, pAcceptor);
    }
    pModel->switchSubGraph("main");

    return pModel;
}

void ZsgReader::_parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph)
{

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
    _parseOutputConnections(pAcceptor);

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

    pAcceptor->EndSubgraph();
}

void ZsgReader::_parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor)
{
    const auto& objValue = node.value;
    const rapidjson::Value& nameValue = objValue["name"];
    const QString& name = nameValue.GetString();

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
        if (name == "MakeDict")
        {
            _parseBySocketKeys(inputs, objValue);
        }
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
    if (objValue.HasMember("socket_keys"))
    {
        auto socket_keys = objValue["socket_keys"].GetArray();
        QJsonArray socketKeys;
        QStringList _keys;
        for (int i = 0; i < socket_keys.Size(); i++)
        {
            socketKeys.append(socket_keys[i].GetString());
            _keys.append(socket_keys[i].GetString());
        }
        nodeData[ROLE_SOCKET_KEYS] = socketKeys;

        PARAM_INFO info;
        info.name = "_KEYS";
        info.value = _keys.join("\n");

        PARAMS_INFO params = nodeData[ROLE_PARAMETERS].value<PARAMS_INFO>();
        params.insert(info.name, info);
        nodeData[ROLE_PARAMETERS] = QVariant::fromValue(params);
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

void ZsgReader::_parseOutputConnections(IAcceptor* pAcceptor)
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

void ZsgReader::_parseBySocketKeys(const rapidjson::Value& objValue, IAcceptor* pAcceptor)
{
    auto socket_keys = objValue["socket_keys"].GetArray();
    QJsonArray socketKeys;
    for (int i = 0; i < socket_keys.Size(); i++)
    {
        QString key = socket_keys[i].GetString();
        INPUT_SOCKET socket;
        socket.info.name = key;
        //socket.info.control = 
        inputSocks[socket.info.name] = socket;
    }
}

void ZsgReader::_parseInputs(const NODE_DESCS& descriptors, const QMap<QString, QString>& objId2Name, 
    const rapidjson::Value& inputs, IAcceptor* pAcceptor)
{
    const auto &inputsObj = inputs.GetObject();
    for (INPUT_SOCKET& inputSocket : inputSockets)
    {
        QByteArray bytes = inputSocket.info.name.toUtf8();
        const auto &inputObj = inputsObj[bytes.data()];
        if (inputObj.IsArray())
        {
            QString type = inputSocket.info.type;
            if (type == "NumericObject")
            {
                type = "float";
            }
            if (type.startsWith("enum "))
            {

            }
            else
            {
                static QStringList acceptTypes = {"int", "bool", "float", "string", "writepath", "readpath"};
                if (type.isEmpty() || acceptTypes.indexOf(type) == -1)
                {
                    inputSocket.info.defaultValue = QVariant();
                }
            }

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
            }
            else
            {
            }
        }
    }
}

void ZsgReader::_parseParams(const rapidjson::Value& jsonParams, IAcceptor* pAcceptor)
{
    if (jsonParams.IsObject())
    {
        const auto &paramsObj = jsonParams.GetObject();
        for (PARAM_INFO &param : params)
        {
            const QString &name = param.name;
            QByteArray bytes = name.toUtf8();
            const auto &paramObj = paramsObj[bytes.data()];
            rapidjson::Type type = paramObj.GetType();
            param.bEnableConnect = false;
            param.value = UiHelper::parseVariantValue(paramObj);
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

void ZsgReader::_parseColorRamps(const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor)
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
                const QString& socketDefl = input_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName, QPointF(), true);
                inputSocket.info.type = socketType;
                inputSocket.info.control = _getControlType(socketType);
                inputSocket.info.defaultValue = _parseDefaultValue(socketDefl);
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
                const QString& socketDefl = param_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrlType;
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = _parseDefaultValue(socketDefl);

                desc.params.insert(socketName, paramInfo);
            }
        }

        for (int i = 0; i < outputs.Size(); i++) {
            if (outputs[i].IsArray()) {
                auto output_triple = outputs[i].GetArray();
                const QString& socketType = output_triple[0].GetString();
                const QString& socketName = output_triple[1].GetString();
                const QString& socketDefl = output_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName, QPointF(), false);
                outputSocket.info.type = socketType;
                outputSocket.info.control = _getControlType(socketType);
                outputSocket.info.defaultValue = _parseDefaultValue(socketDefl);

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