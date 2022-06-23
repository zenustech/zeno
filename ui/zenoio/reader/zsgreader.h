#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <QObject>
#include <QtWidgets>

#include <rapidjson/document.h>

#include "../acceptor/iacceptor.h"
#include <zenoui/model/modeldata.h>

class CurveModel;

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    bool openFile(const QString& fn, IAcceptor* pAcceptor);
    static QVariant _parseToVariant(const QString &type, const rapidjson::Value &val, QObject *parentRef);

private:
    ZsgReader();
    bool _parseSubGraph(const QString& name, const rapidjson::Value &subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseInputs(const QString& id, const QString& nodeName, const NODE_DESCS& descriptors, const rapidjson::Value& inputs, IAcceptor* pAcceptor);
    void _parseParams(const QString& id, const rapidjson::Value &jsonParams, IAcceptor* pAcceptor);
    void _parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor);
    void _parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    void _parseDictKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    QVariant _parseDefaultValue(const QString& val, const QString &type);
    static CurveModel* _parseCurveModel(const rapidjson::Value& jsonCurve, QObject* parentRef);
    NODE_DESCS _parseDescs(const rapidjson::Value& descs);
};

#endif
