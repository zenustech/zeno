#ifndef __TRANSFER_ACCEPTOR_H__
#define __TRANSFER_ACCEPTOR_H__

#include <zenoio/acceptor/iacceptor.h>

class TransferAcceptor : public IAcceptor
{
public:
    TransferAcceptor(IGraphsModel* pModel);

    //IAcceptor
    void setLegacyDescs(const rapidjson::Value &graphObj, const NODE_DESCS &legacyDescs) override;
    void BeginSubgraph(const QString &name) override;
    void EndSubgraph() override;
    bool setCurrentSubGraph(IGraphsModel *pModel, const QModelIndex &subgIdx) override;
    void setFilePath(const QString &fileName) override;
    void switchSubGraph(const QString &graphName) override;
    bool addNode(const QString &nodeid, const QString &name, const NODE_DESCS &descriptors) override;
    void setViewRect(const QRectF &rc) override;
    void setSocketKeys(const QString &id, const QStringList &keys) override;
    void initSockets(const QString &id, const QString &name, const NODE_DESCS &legacyDescs) override;
    void addDictKey(const QString &id, const QString &keyName, bool bInput) override;
    void setInputSocket(const QString &nodeCls, const QString &id, const QString &inSock, const QString &outId,
                        const QString &outSock, const rapidjson::Value &defaultValue,
                        const NODE_DESCS &legacyDescs) override;
    void setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value) override;
    void setPos(const QString &id, const QPointF &pos) override;
    void setOptions(const QString &id, const QStringList &options) override;
    void setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps) override;
    void setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard) override;
    void setLegacyCurve(
        const QString& id,
        const QVector<QPointF>& pts,
        const QVector<QPair<QPointF, QPointF>>& hdls) override;
    QObject *currGraphObj() override;
    void endInputs() override;

    //TransferAcceptor
    QMap<QString, NODE_DATA> nodes() const;
    void getDumpData(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links);
    void reAllocIdents();

private:
    IGraphsModel* m_pModel;
    QMap<QString, NODE_DATA> m_nodes;
    QList<EdgeInfo> m_links;
};


#endif
