#ifndef __IACCEPTOR_H__
#define __IACCEPTOR_H__

#include <zenoui/model/modeldata.h>

class IAcceptor
{
public:
    virtual void setDescriptors(const NODE_DESCS& nodesParams) = 0;
    virtual void BeginSubgraph(const QString& name) = 0;
    virtual void EndSubgraph() = 0;
    virtual void setFilePath(const QString& fileName) = 0;
    virtual void switchSubGraph(const QString& graphName) = 0;
    virtual void addNode(const QString& nodeid, const QString& name, const NODE_DESCS& descriptors) = 0;
    virtual void setViewRect(const QRectF& rc) = 0;
    virtual void setSocketKeys(const QString& id, const QStringList& keys) = 0;
    virtual void initSockets(const QString& id, const QString& name, const NODE_DESCS& descs) = 0;
    virtual void setInputSocket(const QString& nodeCls, const QString& id, const QString& inSock, const QString& outId, const QString& outSock, const QVariant& defaultValue) = 0;
    virtual void setParamValue(const QString& id, const QString& name, const QVariant& var) = 0;
    virtual void setPos(const QString& id, const QPointF& pos) = 0;
    virtual void setOptions(const QString& id, const QStringList& options) = 0;
    virtual void setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps) = 0;
    virtual void setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard) = 0;
    virtual QObject* currGraphObj() = 0;
    virtual ~IAcceptor() = default;
};


#endif
