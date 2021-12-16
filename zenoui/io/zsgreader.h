#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include "../model/nodeitem.h"
#include "../model/graphsmodel.h"
#include "../model/subgraphmodel.h"
#include "../model/modeldata.h"

class NodeItem;
class NodesModel;

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    GraphsModel* loadZsgFile(const QString& fn);

private:
    ZsgReader();
    SubGraphModel* _parseSubGraph(const NODES_PARAMS& pGraphs, const rapidjson::Value &subgraph);
    void _parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph);
    void _parseInputs(INPUT_SOCKETS& inputSocks, const rapidjson::Value& inputs);
    void _parseNodes(const rapidjson::Value& inputs, NODES_PARAMS& nodesDict);
    void _parseParams(PARAMS_INFO &params, const rapidjson::Value &jsonParams);
    void _parseOutputConnections(SubGraphModel* pModel);
    QVariant _parseDefaultValue(const QString& val);
    PARAM_CONTROL _getControlType(const QString& type);
};

#endif