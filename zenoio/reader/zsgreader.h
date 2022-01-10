#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <rapidjson/document.h>
#include <QtWidgets>
#include "acceptor/iacceptor.h"
#include <model/modeldata.h>

class NodeItem;
class NodesModel;

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    void loadZsgFile(const QString& fn, IAcceptor* pAcceptor);
    QString dumpNodeData(const NODE_DATA& data);
    NODE_DATA importNodeData(const QString json);

private:
    ZsgReader();
    void _parseSubGraph(const QString& name, const rapidjson::Value &subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseGraph(NodesModel *pModel, const rapidjson::Value &subgraph);
    void _parseInputs(const NODE_DESCS& descriptors, const QMap<QString, QString>& objId2Name, const rapidjson::Value& inputs, IAcceptor* pAcceptor);
    void _parseParams(const rapidjson::Value &jsonParams, IAcceptor* pAcceptor);
    void _parseColorRamps(const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor);
    void _parseBySocketKeys(const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    void _parseOutputConnections(IAcceptor* pAcceptor);
    QVariant _parseDefaultValue(const QString& val);
    PARAM_CONTROL _getControlType(const QString& type);
    NODE_DESCS _parseDescs(const rapidjson::Value& descs);
};

#endif