#ifndef __IACCEPTOR_H__
#define __IACCEPTOR_H__

#include <rapidjson/document.h>
#include "common.h"
#include <zenomodel/include/modeldata.h>
#include <zenoio/include/common.h>

class IGraphsModel;

class IAcceptor
{
public:
    virtual bool setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS &legacyDescs) = 0;
    virtual void BeginSubgraph(const QString& name, int type, bool bForkLocked) = 0;
    virtual void EndSubgraph() = 0;
    virtual void EndGraphs() = 0;
    virtual bool setCurrentSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx) = 0;
    virtual void setFilePath(const QString& fileName) = 0;
    virtual void switchSubGraph(const QString& graphName) = 0;
    virtual bool addNode(QString& nodeid, const QString& name, const QString& customName, const NODE_DESCS& descriptors) = 0;
    virtual void setViewRect(const QRectF& rc) = 0;
    virtual void setSocketKeys(const QString& id, const QStringList& keys) = 0;
    virtual void initSockets(const QString& id, const QString& name, const NODE_DESCS& legacyDescs) = 0;
    virtual void addDictKey(const QString& id, const QString& keyName, bool bInput) = 0;

    virtual void addSocket(bool bInput, const QString& ident, const QString& sockName, const QString& sockProperty) = 0;

    //legacy:
    virtual void setInputSocket(
        const QString& nodeCls,
        const QString& inNode,
        const QString& inSock,
        const QString& outNode,
        const QString& outSock,
        const rapidjson::Value& defaultValue,
        const NODE_DESCS& legacyDescs
    ) = 0;
    //new socket io format process:
    virtual void setInputSocket2(
        const QString& nodeCls,
        const QString& inNode,
        const QString& inSock,
        const QString& outLinkPath,
        const QString& sockProperty,
        const rapidjson::Value& defaultValue,
        const NODE_DESCS& legacyDescs
    ) = 0;

    virtual void setOutputSocket(
        const QString& inNode,
        const QString& inSock,
        const QString& netlabel,
        const QString& type
    ) = 0;

    virtual void addInnerDictKey(
            bool bInput,
            const QString& ident,
            const QString& sockName,
            const QString& keyName,
            const QString& link,
            const QString& netLabel
    ) = 0;
    virtual void setDictPanelProperty(
            bool bInput,
            const QString& ident,
            const QString& sockName,
            bool bCollasped
    ) = 0;

	virtual void setControlAndProperties(const QString& nodeCls, const QString& inNode, const QString& inSock, PARAM_CONTROL control, const QVariant& ctrlProperties) = 0;
    virtual void setToolTip(PARAM_CLASS cls, const QString& inNode, const QString& inSock, const QString& toolTip) = 0;
    virtual void setNetLabel(PARAM_CLASS cls, const QString& inNode, const QString& inSock, const QString& netlabel) = 0;

    virtual void endInputs(const QString& id, const QString& nodeCls) = 0;
    virtual void setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value, const NODE_DESCS& legacyDescs) = 0;
    virtual void setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params) = 0;
    virtual void endParams(const QString& id, const QString& nodeCls) = 0;
    virtual void setPos(const QString& id, const QPointF& pos) = 0;
    virtual void setOptions(const QString& id, const QStringList& options) = 0;
    virtual void setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps) = 0;
    virtual void setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard) = 0;
    virtual void setTimeInfo(const TIMELINE_INFO& info) = 0;
    virtual void setRecordInfo(const RECORD_SETTING& info) = 0;
    virtual void setLayoutInfo(const LAYOUT_SETTING& info) = 0;
    virtual void setUserDataInfo(const USERDATA_SETTING& info) = 0;
    virtual TIMELINE_INFO timeInfo() const = 0;
    virtual RECORD_SETTING recordInfo() const = 0;
    virtual LAYOUT_SETTING layoutInfo() const = 0;
    virtual USERDATA_SETTING userdataInfo() const = 0;
    virtual void setLegacyCurve(
        const QString& id,
        const QVector<QPointF>& pts,
        const QVector<QPair<QPointF, QPointF>>& hdls) = 0;
    virtual QObject* currGraphObj() = 0;
    virtual void addCustomUI(const QString& id, const VPARAM_INFO& invisibleRoot) = 0;
    virtual void setIOVersion(zenoio::ZSG_VERSION version) = 0;
    virtual void endNode(const QString& id, const QString& nodeCls, const rapidjson::Value& objValue) = 0;
    virtual void addCommandParam(const rapidjson::Value& val, const QString& path) {};
    virtual ~IAcceptor() = default;
};


#endif
