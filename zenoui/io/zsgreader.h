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
    SubGraphModel* _parseSubGraph(GraphsModel* pGraphsModel, const rapidjson::Value &subgraph);
    void _parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph);
    void _parseInputs(INPUT_SOCKETS& inputSocks, const rapidjson::Value& inputs);
    void _parseParams(PARAMS_INFO &params, const rapidjson::Value &jsonParams);
    void _parseColorRamps(COLOR_RAMPS& colorRamps, const rapidjson::Value& jsonColorRamps);
    void _parseOutputConnections(SubGraphModel* pModel);
    QVariant _parseDefaultValue(const QString& val);
    PARAM_CONTROL _getControlType(const QString& type);
};

#endif