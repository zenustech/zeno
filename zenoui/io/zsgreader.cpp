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

    NODE_DESCS nodesDescs = UiHelper::parseDescs(doc["descs"]);
    pModel->setDescriptors(nodesDescs);

    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        SubGraphModel *subGraphModel = _parseSubGraph(pModel, subgraph.value);
        subGraphModel->setName(graphName);

        QStandardItem *pItem = new QStandardItem;
        QVariant var(QVariant::fromValue(static_cast<void*>(subGraphModel)));
        pItem->setText(graphName);
        pItem->setData(var, ROLE_GRAPHPTR);
        pItem->setData(graphName, ROLE_OBJNAME);
        pModel->appendRow(pItem);
    }
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

    for (const auto& node : nodes.GetObject())
    {
        NODEITEM_PTR pItem(new PlainNodeItem);

        const QString& nodeid = node.name.GetString();
        pItem->setData(nodeid, ROLE_OBJID);

        const auto &objValue = node.value;
        const rapidjson::Value& nameValue = objValue["name"];
        const QString &name = nameValue.GetString();

        if (descriptors.find(name) == descriptors.end())
        {
            qDebug() << QString("no node class named [%1]").arg(name);
            continue;
        }

        pItem->setData(nameValue.GetString(), ROLE_OBJNAME);
        pItem->setData(NORMAL_NODE, ROLE_OBJTYPE);

        if (objValue.HasMember("inputs"))
        {
            INPUT_SOCKETS inputs = descriptors[name].inputs;
            _parseInputs(inputs, objValue["inputs"]);
            pItem->setData(QVariant::fromValue(inputs), ROLE_INPUTS);
        }

        OUTPUT_SOCKETS outputs = descriptors[name].outputs;
        pItem->setData(QVariant::fromValue(outputs), ROLE_OUTPUTS);

        if (objValue.HasMember("params"))
        {
            PARAMS_INFO params = descriptors[name].params;
            _parseParams(params, objValue["params"]);
            pItem->setData(QVariant::fromValue(params), ROLE_PARAMETERS);
        }
        if (objValue.HasMember("uipos"))
        {
            auto uipos = objValue["uipos"].GetArray();
            pItem->setData(QPointF(uipos[0].GetFloat(), uipos[1].GetFloat()), ROLE_OBJPOS);
        }
        if (objValue.HasMember("color_ramps"))
        {
            COLOR_RAMPS colorRamps;
            _parseColorRamps(colorRamps, objValue["color_ramps"]);
            pItem->setData(HEATMAP_NODE, ROLE_OBJTYPE);
            pItem->setData(QVariant::fromValue(colorRamps), ROLE_COLORRAMPS);
        }
        if (name == "Blackboard")
        {
            pItem->setData(BLACKBOARD_NODE, ROLE_OBJTYPE);
            if (objValue.HasMember("special"))
            {
                pItem->setData(objValue["special"].GetBool(), ROLE_BLACKBOARD_SPECIAL);
            }
            pItem->setData(objValue.HasMember("title") ? objValue["title"].GetString() : "", ROLE_BLACKBOARD_TITLE);
            pItem->setData(objValue.HasMember("content") ? objValue["content"].GetString() : "", ROLE_BLACKBOARD_CONTENT);
            if (objValue.HasMember("width") && objValue.HasMember("height"))
            {
                qreal w = objValue["width"].GetFloat();
                qreal h = objValue["height"].GetFloat();
                pItem->setData(QSizeF(w, h), ROLE_BLACKBOARD_SIZE);
            }
            if (objValue.HasMember("params"))
            {
                //todo
            }
        }
        pModel->appendItem(pItem);
    }
    _parseOutputConnections(pModel);

    //view rect
    QRectF viewRect;
    if (subgraph.HasMember("view_rect")) {
        const auto &obj = subgraph["view_rect"];
        viewRect = QRectF(obj["x"].GetFloat(), obj["y"].GetFloat(), obj["width"].GetFloat(), obj["height"].GetFloat());
    }
    pModel->setViewRect(viewRect);

    return pModel;
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

void ZsgReader::_parseInputs(INPUT_SOCKETS& inputSockets, const rapidjson::Value& inputs)
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
                const QString &outId = arr[0].GetString();
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