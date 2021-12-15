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

    //todo: smart pointer
    GraphsModel* pModel = new GraphsModel;
    for (const auto& subgraph : graph.GetObject())
    {
        const QString& graphName = subgraph.name.GetString();
        SubGraphModel* subGraphModel = _parseSubGraph(subgraph.value);
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

SubGraphModel* ZsgReader::_parseSubGraph(const rapidjson::Value &subgraph)
{
    //todo: should consider descript info. some info of outsock without connection show in descript info.

    SubGraphModel* pModel = new SubGraphModel;
    const auto& nodes = subgraph["nodes"];
    if (nodes.IsNull())
        return nullptr;

    for (const auto& node : nodes.GetObject())
    {
        NODEITEM_PTR pItem(new PlainNodeItem);

        pItem->setData(node.name.GetString(), ROLE_OBJID);

        const auto &objValue = node.value;
        const rapidjson::Value& nameValue = objValue["name"];
        const QString &name = nameValue.GetString();
        pItem->setData(nameValue.GetString(), ROLE_OBJNAME);

        const INPUT_SOCKETS &inputs = _parseInputs(objValue["inputs"]);
        pItem->setData(QVariant::fromValue(inputs), ROLE_INPUTS);
        pItem->setData(QVariant::fromValue(OUTPUT_SOCKETS()), ROLE_OUTPUTS);

        const rapidjson::Value& params = objValue["params"];
        pItem->setData(QVariant::fromValue(_parseParams(inputs, params)), ROLE_PARAMETERS);

        const rapidjson::Value& uipos = objValue["uipos"];
        auto pos = uipos.GetArray();
        pItem->setData(QPointF(pos[0].GetFloat(), pos[1].GetFloat()), ROLE_OBJPOS);

        pModel->appendItem(pItem);
    }
    _parseOutputs(pModel);
    return pModel;
}

void ZsgReader::_parseOutputs(SubGraphModel* pModel)
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

INPUT_SOCKETS ZsgReader::_parseInputs(const rapidjson::Value& inputs)
{
    INPUT_SOCKETS inputSockets;
    for (const auto &inSockInfo : inputs.GetObject())
    {
        INPUT_SOCKET inputSocket;
        inputSocket.info.name = inSockInfo.name.GetString();
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
                inputSocket.defaultValue = arr[2].GetString();
            } else if (arr[2].GetType() == rapidjson::kNumberType) {
                inputSocket.defaultValue = arr[2].GetFloat();
            }
        }
        inputSockets.insert(inputSocket.info.name, inputSocket);
    }
    return inputSockets;
}

PARAMS_INFO ZsgReader::_parseParams(const INPUT_SOCKETS& inputs, const rapidjson::Value &jsonParams)
{
    PARAMS_INFO params;
    for (const auto &jsonParam : jsonParams.GetObject())
    {
        PARAM_INFO param;
        param.bEnableConnect = false;

        const QString& name = jsonParam.name.GetString();
        rapidjson::Type type = jsonParam.value.GetType();
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
        params.insert(name, param);
    }
    for (auto inSock : inputs)
    {
        PARAM_INFO param;
        param.defaultValue = inSock.defaultValue;
        param.bEnableConnect = true;
        param.name = inSock.info.name;
        param.value = QVariant();   //current set to null when no calc started.
        params.insert(param.name, param);
    }
    return params;
}