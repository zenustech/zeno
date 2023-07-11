#ifndef __ZSG_READER_H__
#define __ZSG_READER_H__

#include <QObject>
#include <QtWidgets>

#include <rapidjson/document.h>
#include "common.h"
#include <zenomodel/include/modeldata.h>
#include <zenoio/include/common.h>
#include <zenomodel/include/modeldata.h>

class IGraphsModel;

namespace zenoio
{
    class ZsgReader
    {
    public:
        static ZsgReader& getInstance();
        bool openFile(const QString& fn, ZSG_PARSE_RESULT& ret);
        bool openSubgraphFile(const QString& fn, ZSG_PARSE_RESULT& ret);
        bool importNodes(
                IGraphsModel* pModel,
                const QModelIndex& subgIdx,
                const QString& nodeJson,
                const QPointF& targetPos,
                SUBGRAPH_DATA& subgraph);

    private:
        ZsgReader();
        bool _parseSubGraph(
                const QString& name,
                const rapidjson::Value &subgraph,
                const NODE_DESCS& descriptors,
                const QMap<QString, SUBGRAPH_DATA>& subgraphDatas,
                SUBGRAPH_DATA& subgData);

        bool _parseNode(
                const QString& subgPath,
                const QString& nodeid,
                const rapidjson::Value& nodeObj,
                const NODE_DESCS& descriptors,
                const QMap<QString, SUBGRAPH_DATA>& subgraphDatas,
                NODE_DATA& ret,
                LINKS_DATA& links);

        void _parseSocket(
                const QString& subgPath,
                const QString& id,
                const QString& nodeName,
                const QString& inSock,
                bool bInput,
                const rapidjson::Value& sockObj,
                const NODE_DESCS& descriptors,
                NODE_DATA& ret,
                LINKS_DATA& links);

        void _parseInputs(
                const QString& subgPath,
                const QString& id,
                const QString& nodeName,
                const NODE_DESCS& descriptors,
                const rapidjson::Value& inputs,
                NODE_DATA& ret,
                LINKS_DATA& links);

        void _parseParams(
                const QString& id,
                const QString& nodeName,
                const rapidjson::Value &jsonParams,
                const NODE_DESCS& legacyDescs,
                NODE_DATA& ret);

        bool _parseParams2(
                const QString& id,
                const QString& nodeCls,
                const rapidjson::Value& jsonParams,
                NODE_DATA& ret);

        void _parseOutputs(
                const QString& id,
                const QString& nodeName,
                const rapidjson::Value& jsonParams,
                NODE_DATA& ret);

        void _parseCustomPanel(
                const QString& id,
                const QString& nodeName,
                const rapidjson::Value& jsonCutomUI,
                NODE_DATA& ret);

        void _parseDictPanel(
                const QString& subgPath,
                bool bInput,
                const rapidjson::Value& dictPanelObj,
                const QString& id,
                const QString& inSock,
                const QString& nodeName,
                NODE_DATA& ret,
                LINKS_DATA& links);

        void _parseColorRamps(const QString& id, const rapidjson::Value& jsonColorRamps, NODE_DATA& ret);
        void _parseLegacyCurves(const QString &id, const rapidjson::Value &jsonPoints, const rapidjson::Value &jsonHandlers,
                                NODE_DATA& ret);
        void _parseViews(const rapidjson::Value& jsonViews, ZSG_PARSE_RESULT& res);
        void _parseTimeline(const rapidjson::Value& jsonTimeline, ZSG_PARSE_RESULT& res);

        void _parseBySocketKeys(const QString& id, const rapidjson::Value& objValue, NODE_DATA& ret);
        void _parseDictKeys(const QString& id, const rapidjson::Value& objValue, NODE_DATA& ret);
        void _parseChildNodes(const QString& id, const rapidjson::Value& jsonNodes, const NODE_DESCS& descriptors, NODE_DATA& ret);
        NODE_DESCS _parseDescs(const rapidjson::Value& descs);

        NODES_DATA _parseChildren(const rapidjson::Value& jsonNodes);
        void initSockets(const QString& name, const NODE_DESCS& legacyDescs, NODE_DATA& ret);

        QVariant _parseDeflValue(
                        const QString &nodeCls,
                        const NODE_DESCS &legacyDescs,
                        const QString& sockName,
                        PARAM_CLASS cls,
                        const rapidjson::Value &defaultValue);

        ZSG_VERSION m_ioVer;
        bool m_bDiskReading;        //disk io read.
    };
}

#endif
