#ifndef __TREEMODEL_ACCEPTOR_H__
#define __TREEMODEL_ACCEPTOR_H__

#include <zenoio/acceptor/iacceptor.h>
#include "modelacceptor.h"
#include "modeldata.h"

class GraphsModel;
class GraphsTreeModel;
class SubGraphModel;

class TreeAcceptor : public IAcceptor
{
public:
    TreeAcceptor(GraphsTreeModel* pModel, GraphsModel* pSubgraphs, bool bImport);

    //IAcceptor
    bool setLegacyDescs(const rapidjson::Value &graphObj, const NODE_DESCS &nodesParams) override;
    void BeginSubgraph(const QString &name) override;
    bool setCurrentSubGraph(IGraphsModel *pModel, const QModelIndex &subgIdx) override;
    void EndSubgraph() override;
    void EndGraphs() override;
    void switchSubGraph(const QString &graphName) override;
    bool addNode(const QString &nodeid, const QString &name, const QString &customName,
                 const NODE_DESCS &descriptors) override;
    void setViewRect(const QRectF &rc) override;
    void setSocketKeys(const QString& nodePath, const QStringList& keys) override;
    void initSockets(const QString& nodePath, const QString& name, const NODE_DESCS& descs) override;
    void addDictKey(const QString& nodePath, const QString& keyName, bool bInput) override;
    void addSocket(bool bInput, const QString& nodePath, const QString& sockName, const QString& sockProperty) override;

    void setInputSocket2(
            const QString &nodeCls,
            const QString &inNode,
            const QString &inSock,
            const QString &outLinkPath,
            const QString &sockProperty,
            const rapidjson::Value &defaultValue) override;

    void setInputSocket(
            const QString &nodeCls,
            const QString &inNode,
            const QString &inSock,
            const QString &outNode,
            const QString &outSock,
            const rapidjson::Value &defaultValue) override;

    void addInnerDictKey(bool bInput, const QString &inNode, const QString &inSock, const QString &keyName,
                         const QString &link) override;

    void setDictPanelProperty(bool bInput, const QString &ident, const QString &sockName, bool bCollasped) override;

    void setControlAndProperties(const QString &nodeCls, const QString &inNode, const QString &inSock,
                                 PARAM_CONTROL control, const QVariant &ctrlProperties);
    void setToolTip(PARAM_CLASS cls, const QString &inNode, const QString &inSock, const QString &toolTip) override;

    void setParamValue(const QString &id, const QString &nodeCls, const QString &name,
                       const rapidjson::Value &value) override;
    void setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params) override;
    void setPos(const QString &id, const QPointF &pos) override;
    void setOptions(const QString &id, const QStringList &options) override;
    void setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps) override;
    void setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard) override;
    void setTimeInfo(const TIMELINE_INFO &info) override;
    TIMELINE_INFO timeInfo() const override;
    void setLegacyCurve(const QString &id, const QVector<QPointF> &pts,
                        const QVector<QPair<QPointF, QPointF>> &hdls) override;
    QObject *currGraphObj() override;
    void endInputs(const QString &id, const QString &nodeCls) override;
    void endParams(const QString &id, const QString &nodeCls) override;
    void addCustomUI(const QString &id, const VPARAM_INFO &invisibleRoot) override;
    void setIOVersion(zenoio::ZSG_VERSION versio) override;
    void resolveAllLinks() override;

private:
    QModelIndex _getNodeIdx(const QString& identOrObjPath);
    QModelIndex _getSockIdx(const QString& inNode, const QString& sockName, bool bInput);

    TIMELINE_INFO m_timeInfo;
    GraphsTreeModel* m_pNodeModel;
    GraphsModel* m_pSubgraphs;
    std::shared_ptr<ModelAcceptor> m_pSubgAcceptor;
    QList<EdgeInfo> m_links;
    zenoio::ZSG_VERSION m_ioVer;
    bool m_bImport;
    bool m_bImportMain;        //whether importing main Subgraph right now
};


#endif