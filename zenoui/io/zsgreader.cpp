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

        const rapidjson::Value& inputs = objValue["inputs"];
        const QJsonObject& inputsObj = _parseInputs(inputs);
        pItem->setData(inputsObj, ROLE_INPUTS);

        const rapidjson::Value& params = objValue["params"];
        const QJsonObject& paramsObj = _parseParams(params);
        pItem->setData(paramsObj, ROLE_PARAMETERS);

        const rapidjson::Value& uipos = objValue["uipos"];
        auto pos = uipos.GetArray();
        pItem->setData(QPointF(pos[0].GetFloat(), pos[1].GetFloat()), ROLE_OBJPOS);

        pModel->appendItem(pItem);
    }

    //init output ports for each node.
    int n = pModel->rowCount();
    for (int r = 0; r < n; r++)
    {
        const QModelIndex &idx = pModel->index(r, 0);
        const QString& id = idx.data(ROLE_OBJID).toString();
        const QJsonObject& inputs = idx.data(ROLE_INPUTS).toJsonObject();
        std::map<QString, QJsonObject> outputs;
        foreach (const QString& key, inputs.keys())
        {
            QJsonValue val = inputs.value(key);
            Q_ASSERT(val.type() == rapidjson::kArrayType);
            QJsonArray arr = val.toArray();
            Q_ASSERT(arr.size() == 3);
            const QString& fromId = arr[0].isString() ? arr[0].toString() : 0;
            const QString& portName = arr[1].isString() ? arr[1].toString() : 0;
            if (fromId.isEmpty() || portName.isEmpty())
                continue;

            arr[0] = id;
            arr[1] = key;
            outputs[fromId].insert(portName, arr);
        }
        for (auto it : outputs)
        {
            const QString &nodeid = it.first;
            const QJsonObject &obj = it.second;
            const QModelIndex &fromIndex = pModel->index(nodeid);
            pModel->setData(fromIndex, obj, ROLE_OUTPUTS);
        }
    }
    return pModel;
}

QJsonObject ZsgReader::_parseInputs(const rapidjson::Value& inputs)
{
    QJsonObject jsonInputs;
    for (const auto &node : inputs.GetObject())
    {
        QJsonObject jsonPort;
        const QString& name = node.name.GetString();
        const auto& arr = node.value.GetArray();
        
        QJsonArray jsonArr;
        RAPIDJSON_ASSERT(arr.Size() == 3);
        const QString& objId = arr[0].IsNull() ? "" : arr[0].GetString();
        const QString& outPort = arr[1].IsNull() ? "" : arr[1].GetString();
        jsonArr.append(objId);
        jsonArr.append(outPort);

        rapidjson::Type type = arr[2].GetType();
        if (type == rapidjson::kNullType) {
            jsonArr.append("");
        } else if (type == rapidjson::kStringType) {
            jsonArr.append(arr[2].GetString());
        } else if (type == rapidjson::kNumberType) {
            jsonArr.append(arr[2].GetFloat());
        } else {
            jsonArr.append("");
        }

        jsonInputs.insert(name, jsonArr);
    }
    return jsonInputs;
}

QJsonObject ZsgReader::_parseParams(const rapidjson::Value& params)
{
    QJsonObject jsonParams;
    for (const auto &param : params.GetObject())
    {
        const QString& name = param.name.GetString();
        rapidjson::Type type = param.value.GetType();
        if (type == rapidjson::kNullType)
        {
            jsonParams.insert(name, "");
        }
        else if (type == rapidjson::kStringType)
        {
            jsonParams.insert(name, param.value.GetString());
        }
        else if (type == rapidjson::kNumberType)
        {
            jsonParams.insert(name, param.value.GetDouble());
        }
    }
    return jsonParams;
}