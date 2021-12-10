#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include "../model/nodeitem.h"
#include "../model/graphsmodel.h"
#include "../model/subgraphmodel.h"

class NodeItem;
class NodesModel;

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    GraphsModel* loadZsgFile(const QString& fn);

private:
    ZsgReader();
    SubGraphModel* _parseSubGraph(const rapidjson::Value &subgraph);
    void _parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph);
    QJsonObject _parseInputs(const rapidjson::Value &inputs);
    QJsonObject _parseParams(const rapidjson::Value &params);
};

#endif