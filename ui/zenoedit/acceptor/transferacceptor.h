#ifndef __TRANSFER_ACCEPTOR_H__
#define __TRANSFER_ACCEPTOR_H__

#include <zenoio/acceptor/iacceptor.h>

class TransferAcceptor : public IAcceptor
{
public:
    TransferAcceptor(IGraphsModel* pModel);

    //IAcceptor
    void setLegacyDescs(const rapidjson::Value &graphObj, const NODE_DESCS &legacyDescs);
    void BeginSubgraph(const QString &name);
    void EndSubgraph();
    bool setCurrentSubGraph(IGraphsModel *pModel, const QModelIndex &subgIdx);
    void setFilePath(const QString &fileName);
    void switchSubGraph(const QString &graphName);
    bool addNode(const QString &nodeid, const QString &name, const NODE_DESCS &descriptors);
    void setViewRect(const QRectF &rc);
    void setSocketKeys(const QString &id, const QStringList &keys);
    void initSockets(const QString &id, const QString &name, const NODE_DESCS &legacyDescs);
    void addDictKey(const QString &id, const QString &keyName, bool bInput);
    void setInputSocket(const QString &nodeCls, const QString &id, const QString &inSock, const QString &outId,
                        const QString &outSock, const rapidjson::Value &defaultValue,
                        const NODE_DESCS &legacyDescs);
    void setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value);
    void setPos(const QString &id, const QPointF &pos);
    void setOptions(const QString &id, const QStringList &options);
    void setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps);
    void setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard);
    QObject *currGraphObj();

    //TransferAcceptor
    QMap<QString, NODE_DATA> nodes() const;

private:
    IGraphsModel* m_pModel;
    QMap<QString, NODE_DATA> m_nodes;
};


#endif