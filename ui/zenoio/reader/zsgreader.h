#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <QObject>
#include <QtWidgets>

#include <rapidjson/document.h>

#include "../acceptor/iacceptor.h"
#include <zenomodel/include/modeldata.h>

class CurveModel;
class IGraphsModel;

class ZsgReader
{
public:
    static ZsgReader& getInstance();
    bool openFile(const QString& fn, IAcceptor* pAcceptor);
    bool parseZsg(const QString& fn, const QByteArray& bytes, IAcceptor* pAcceptor);
    bool importSubgraphs(const QString& fn, IAcceptor* pAcceptor, const QMap<QString, QString>& graphs, IGraphsModel* pModel);
    bool importNodes(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& nodeJson, const QPointF& targetPos, IAcceptor* pAcceptor);
    PtrLayoutNode _readLayout(const rapidjson::Value& objValue);
    bool readRenderSettings(const QString& fn, ZENO_RECORD_RUN_INITPARAM& param);

private:
    ZsgReader();
    bool _parseSubGraph(const QString& name, const rapidjson::Value &subgraph, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    bool _parseNode(const QString& nodeid, const rapidjson::Value& nodeObj, const NODE_DESCS& descriptors, IAcceptor* pAcceptor);
    void _parseSocket(const QString& id, const QString& nodeName, const NODE_DESCS& descriptors, const QString& inSock, bool bInput, const rapidjson::Value& sockObj, IAcceptor* pAcceptor);
    void _parseInputs(const QString& id, const QString& nodeName, const NODE_DESCS& descriptors, const rapidjson::Value& inputs, IAcceptor* pAcceptor);
    void _parseParams(const QString& id, const QString& nodeName, const NODE_DESCS& descriptors, const rapidjson::Value &jsonParams, IAcceptor* pAcceptor);
    bool _parseParams2(const QString& id, const QString &nodeCls, const rapidjson::Value &jsonParams, IAcceptor *pAcceptor);
    void _parseOutputs(const QString& id, const QString& nodeName,const rapidjson::Value &jsonParams, IAcceptor* pAcceptor);
    void _parseCustomPanel(const QString& id, const QString& nodeName, const rapidjson::Value& jsonCutomUI, IAcceptor* pAcceptor);
    void _parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, IAcceptor* pAcceptor);
    void _parseCurvePoints(const QString& id, const rapidjson::Value& jsonPoints, IAcceptor* pAcceptor);
    void _parseCurveHandlers(const QString& id, const rapidjson::Value& jsonHandlers, IAcceptor* pAcceptor);
    void _parseLegacyCurves(const QString &id, const rapidjson::Value &jsonPoints, const rapidjson::Value &jsonHandlers,
                            IAcceptor *pAcceptor);
    void _parseViews(const rapidjson::Value& jsonViews, IAcceptor* pAcceptor);
    void _parseTimeline(const rapidjson::Value& jsonTimeline, IAcceptor* pAcceptor);
    void _parseSettings(const rapidjson::Value& jsonSettings, IAcceptor* pAcceptor);
    void _parseDictPanel(bool bInput, const rapidjson::Value& dictPanelObj, const QString& id, const QString& inSock, const QString& nodeName, IAcceptor* pAcceptor);
    void _parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    void _parseDictKeys(const QString& id, const rapidjson::Value& objValue, IAcceptor* pAcceptor);
    NODE_DESCS _parseDescs(const rapidjson::Value& descs, IAcceptor *pAcceptor);
    void _parseCommands(const rapidjson::Value& jsonViews, IAcceptor* pAcceptor);
};

#endif
