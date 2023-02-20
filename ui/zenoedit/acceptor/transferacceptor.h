#ifndef __TRANSFER_ACCEPTOR_H__
#define __TRANSFER_ACCEPTOR_H__

#include <zenoio/acceptor/iacceptor.h>

class TransferAcceptor : public IAcceptor
{
public:
    TransferAcceptor(IGraphsModel* pModel);

    //IAcceptor
    bool setLegacyDescs(const rapidjson::Value &graphObj, const NODE_DESCS &legacyDescs) override;
    void BeginSubgraph(const QString &name) override;
    void EndSubgraph() override;
    void EndGraphs() override;
    bool setCurrentSubGraph(IGraphsModel *pModel, const QModelIndex &subgIdx) override;
    void setFilePath(const QString &fileName) override;
    void switchSubGraph(const QString &graphName) override;
    bool addNode(const QString &nodeid, const QString &name, const NODE_DESCS &descriptors) override;
    void setViewRect(const QRectF &rc) override;
    void setSocketKeys(const QString &id, const QStringList &keys) override;
    void initSockets(const QString &id, const QString &name, const NODE_DESCS &legacyDescs) override;
    void addDictKey(const QString &id, const QString &keyName, bool bInput) override;
    void addSocket(bool bInput, const QString& ident, const QString& sockName, const QString& sockProperty) override;

    void setInputSocket(
        const QString& nodeCls,
        const QString& inNode,
        const QString& inSock,
        const QString& outNode,
        const QString& outSock,
        const rapidjson::Value& defaultValue
    ) override;

    void setInputSocket2(
                const QString& nodeCls,
                const QString& inNode,
                const QString& inSock,
                const QString& outLinkPath,
                const QString& sockProperty,
                const rapidjson::Value& defaultValue) override;

    void addInnerDictKey(
        bool bInput,
        const QString& inNode,
        const QString& inSock,
        const QString& keyName,
        const QString& link
    ) override;

    void setDictPanelProperty(
        bool bInput,
        const QString& ident,
        const QString& sockName,
        bool bCollasped
    ) override;

	void setControlAndProperties(const QString& nodeCls, const QString& inNode, const QString& inSock, PARAM_CONTROL control, const QVariant& ctrlProperties);
    void setParamValue(const QString& id, const QString& nodeCls, const QString& name, const rapidjson::Value& value) override;
    void setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params) override;
    void setPos(const QString &id, const QPointF &pos) override;
    void setOptions(const QString &id, const QStringList &options) override;
    void setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps) override;
    void setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard) override;
    void setTimeInfo(const TIMELINE_INFO& info) override;
    TIMELINE_INFO timeInfo() const override;
    void setLegacyCurve(
        const QString& id,
        const QVector<QPointF>& pts,
        const QVector<QPair<QPointF, QPointF>>& hdls) override;
    QObject *currGraphObj() override;
    void endInputs(const QString& id, const QString& nodeCls) override;
    void endParams(const QString& id, const QString& nodeCls) override;
    void addCustomUI(const QString& id, bool bNodeUI, const VPARAM_INFO& invisibleRoot) override;

    //TransferAcceptor
    QMap<QString, NODE_DATA> nodes() const;
    QList<EdgeInfo> links() const;
    void getDumpData(QMap<QString, NODE_DATA>& nodes, QList<EdgeInfo>& links);
    void setIOVersion(zenoio::ZSG_VERSION versio) override;

private:
    IGraphsModel* m_pModel;
    QMap<QString, NODE_DATA> m_nodes;
    QList<EdgeInfo> m_links;
    QString m_currSubgraph;
};


#endif
