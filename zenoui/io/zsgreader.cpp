#include "../model/nodesmodel.h"
#include "../model/modelrole.h"
#include "zsgreader.h"


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

    NODE_DESCS nodesDescs;
    _parseDescs(doc["descs"], nodesDescs);
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

        INPUT_SOCKETS inputs = descriptors[name].inputs;
        _parseInputs(inputs, objValue["inputs"]);
        pItem->setData(QVariant::fromValue(inputs), ROLE_INPUTS);

        OUTPUT_SOCKETS outputs = descriptors[name].outputs;
        pItem->setData(QVariant::fromValue(outputs), ROLE_OUTPUTS);

        PARAMS_INFO params = descriptors[name].params;
        _parseParams(params, objValue["params"]);
        pItem->setData(QVariant::fromValue(params), ROLE_PARAMETERS);

        auto uipos = objValue["uipos"].GetArray();
        pItem->setData(QPointF(uipos[0].GetFloat(), uipos[1].GetFloat()), ROLE_OBJPOS);

        pModel->appendItem(pItem);
    }
    _parseOutputConnections(pModel);
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

void ZsgReader::_parseDescs(const rapidjson::Value& descs, NODE_DESCS& nodeDescs)
{
    for (const auto& node : descs.GetObject())
    {
        const QString& name = node.name.GetString();
        const auto& objValue = node.value;
        auto inputs = objValue["inputs"].GetArray();
        auto outputs = objValue["outputs"].GetArray();
        auto params = objValue["params"].GetArray();
        auto categories = objValue["categories"].GetArray();

        NODE_DESC pack;

        for (int i = 0; i < inputs.Size(); i++)
        {
            if (inputs[i].IsArray())
            {
                auto input_triple = inputs[i].GetArray();
                const QString &socketType = input_triple[0].GetString();
                const QString &socketName = input_triple[1].GetString();
                const QString &socketDefl = input_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName, QPointF(), true);
                inputSocket.info.type = socketType;
                inputSocket.info.control = _getControlType(socketType);
                inputSocket.info.defaultValue = _parseDefaultValue(socketDefl);
                pack.inputs.insert(socketName, inputSocket);
            }
            else
            {
            
            }
        }

        for (int i = 0; i < params.Size(); i++)
        {
            if (params[i].IsArray())
            {
                auto param_triple = params[i].GetArray();
                const QString &socketType = param_triple[0].GetString();
                const QString &socketName = param_triple[1].GetString();
                const QString &socketDefl = param_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrlType;
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = _parseDefaultValue(socketDefl);

                pack.params.insert(socketName, paramInfo);
            }
        }

        for (int i = 0; i < outputs.Size(); i++)
        {
            if (outputs[i].IsArray())
            {
                auto output_triple = outputs[i].GetArray();
                const QString &socketType = output_triple[0].GetString();
                const QString &socketName = output_triple[1].GetString();
                const QString &socketDefl = output_triple[2].GetString();
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName, QPointF(), false);
                outputSocket.info.type = socketType;
                outputSocket.info.control = _getControlType(socketType);
                outputSocket.info.defaultValue = _parseDefaultValue(socketDefl);

                pack.outputs.insert(socketName, outputSocket);
            }
            else
            {
            
            }
        }
        nodeDescs.insert(name, pack);
    }
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
    for (const auto &inSockInfo : inputs.GetObject())
    {
        const QString& sockName = inSockInfo.name.GetString();
        INPUT_SOCKET& inputSocket = inputSockets[sockName];

        inputSocket.info.name = sockName;
        const auto& arr = inSockInfo.value.GetArray();
        RAPIDJSON_ASSERT(arr.Size() == 3);

        //only consider one input source, as the form of tuple.
        //for each port. only one port currently.
        if (!arr[0].IsNull())
        {
            const QString &outId = arr[0].GetString();
            if (!arr[1].IsNull())
            {
                const QString socketName = arr[1].GetString();
                inputSocket.outNodes[outId][socketName] = SOCKET_INFO(outId, socketName);
            }
            //to ask: default value type else
            if (arr[2].GetType() == rapidjson::kStringType) {
                inputSocket.info.defaultValue = arr[2].GetString();
            } else if (arr[2].GetType() == rapidjson::kNumberType) {
                inputSocket.info.defaultValue = arr[2].GetFloat();
            }
        }
    }
}

void ZsgReader::_parseParams(PARAMS_INFO& params, const rapidjson::Value& jsonParams)
{
    for (const auto &jsonParam : jsonParams.GetObject())
    {
        //if some param not exists in desc params, should include it?
        const QString& name = jsonParam.name.GetString();
        rapidjson::Type type = jsonParam.value.GetType();

        PARAM_INFO &param = params[name];
        param.bEnableConnect = false;
        if (type == rapidjson::kNullType)
        {
        }
        else if (type == rapidjson::kStringType)
        {
            param.value = jsonParam.value.GetString();
        }
        else if (type == rapidjson::kNumberType)
        {
            param.value = jsonParam.value.GetDouble();
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