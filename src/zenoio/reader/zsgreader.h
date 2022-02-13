#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <QObject>
#include <QtWidgets>

#include <rapidjson/document.h>

#include "../acceptor/iacceptor.h"
#include <zenoui/model/modeldata.h>

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    void loadZsgFile(const QString& fn, IAcceptor* pAcceptor);

private:
    ZsgReader();
    void _parseSubGraph(const QString& name, const rapidjson::Value &subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseInputs(const QString& id, const NODE_DESCS& descriptors, const rapidjson::Value& inputs, IAcceptor* pAcceptor);
    void _parseParams(const QString& id, const rapidjson::Value &jsonParams, IAcceptor* pAcceptor);
    void _parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor);
    void _parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    QVariant _parseDefaultValue(const QString& val, const QString &type);
    QVariant _parseToVariant(const rapidjson::Value& val);
    PARAM_CONTROL _getControlType(const QString& type);
    NODE_DESCS _parseDescs(const rapidjson::Value& descs);
};

#endif
