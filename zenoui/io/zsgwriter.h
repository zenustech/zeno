#ifndef __ZSG_WRITER_H__
#define __ZSG_WRITER_H__

#include "../model/graphsmodel.h"
#include "../model/modeldata.h"
#include "../model/nodeitem.h"
#include "../model/subgraphmodel.h"

class ZsgWriter
{
public:
    static ZsgWriter& getInstance();
    QString dumpProgram(GraphsModel *pModel);
    QString dumpSubGraph(SubGraphModel *pSubModel);
    QJsonObject dumpGraphs(GraphsModel *pMode);
    QJsonObject dumpNode(const NODE_DATA& data);

private:
    ZsgWriter();
    QJsonObject _dumpSubGraph(SubGraphModel *pSubModel);
    QJsonObject _dumpDescriptors(const NODE_DESCS& descs);
};

#endif