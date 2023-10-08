#ifndef __GRAPHICS_TREEMODEL_H__
#define __GRAPHICS_TREEMODEL_H__

#include <QtWidgets>
#include "nodeitem.h"
#include <zenomodel/include/igraphsmodel.h>

class GraphsModel;
class SubGraphModel;
class IGraphsModel;
class GraphsManagment;
class GraphsTreeModel_impl;

class GraphsTreeModel : public IGraphsModel
{
    Q_OBJECT
public:
    GraphsTreeModel(QObject* parent = nullptr);
    ~GraphsTreeModel();

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override;
    QModelIndex index(const QString &subGraphName) const override;
    QModelIndex index(const QString &id, const QModelIndex &subGpIdx) override;
    QModelIndex index(int r, const QModelIndex &subGpIdx) override;
    QModelIndex nodeIndex(const QString &ident) override;
    /* end: node index: */

    QModelIndex nodeIndex(uint32_t sid, uint32_t nodeid) override;
    QModelIndex subgIndex(uint32_t sid) override;

    int itemCount(const QModelIndex &subGpIdx) const override;

    QModelIndex linkIndex(const QModelIndex &subgIdx, int r) override;
    QModelIndex linkIndex(const QModelIndex &subgIdx, const QString &outNode, const QString &outSock,
                                  const QString &inNode, const QString &inSock) override;

    void addNode(const NODE_DATA &nodeData, const QModelIndex &subGpIdx, bool enableTransaction = false) override;
    void setNodeData(const QModelIndex &nodeIndex, const QModelIndex &subGpIdx, const QVariant &value,
                             int role) override;
    void importNodes(const QMap<QString, NODE_DATA> &nodes, const QList<EdgeInfo> &links, const QPointF &pos,
                             const QModelIndex &subGpIdx, bool enableTransaction = false) override;
    void removeNode(const QString &nodeid, const QModelIndex &subGpIdx, bool enableTransaction = false) override;

    QModelIndex addLink(const QModelIndex &subgIdx, const QModelIndex &fromSock, const QModelIndex &toSock,
                                bool enableTransaction = false) override;
    QModelIndex addLink(const QModelIndex &subgIdx, const EdgeInfo &info, bool enableTransaction = false) override;
    void removeLink(const QModelIndex &linkIdx, bool enableTransaction = false) override;
    void removeLink(const QModelIndex &subgIdx, const EdgeInfo &linkIdx, bool enableTransaction = false) override;
    void removeSubGraph(const QString &name) override;
    QModelIndex extractSubGraph(const QModelIndexList &nodes, const QModelIndexList &links,
                                        const QModelIndex &fromSubg, const QString &toSubg,
                                        bool enableTrans = false) override;
    bool IsSubGraphNode(const QModelIndex &nodeIdx) const override;

    /*
	 fork subnet node indexed by subnetNodeIdx under subgIdx. 
	 */
    QModelIndex fork(const QModelIndex &subgIdx, const QModelIndex &subnetNodeIdx) override;

    void updateParamInfo(const QString &id, PARAM_UPDATE_INFO info, const QModelIndex &subGpIdx,
                                 bool enableTransaction = false) override;
    void updateSocketDefl(const QString &id, PARAM_UPDATE_INFO info, const QModelIndex &subGpIdx,
                                  bool enableTransaction = false) override;
    void updateNodeStatus(const QString &nodeid, STATUS_UPDATE_INFO info, const QModelIndex &subgIdx,
                                  bool enableTransaction = false) override;
    void updateBlackboard(const QString &id, const QVariant &blackboard, const QModelIndex &subgIdx,
                                  bool enableTransaction) override;

    NODE_DATA itemData(const QModelIndex &index, const QModelIndex &subGpIdx) const override;
    void exportSubgraph(const QModelIndex& subGpIdx, NODES_DATA& nodes, LINKS_DATA& links) const override;
    void setName(const QString &name, const QModelIndex &subGpIdx) override;

    void clearSubGraph(const QModelIndex &subGpIdx) override;
    void clear() override;
    void undo() override;
    void redo() override;
    void switchSubGraph(const QString &graphName) {
    }
    void newSubgraph(const QString &graphName) override;
    void initMainGraph() override;
    void renameSubGraph(const QString &oldName, const QString &newName) override;
    bool isDirty() const override;
    QModelIndexList searchInSubgraph(const QString &objName, const QModelIndex &idx) override;
    QModelIndexList subgraphsIndice() const override;
    QList<SEARCH_RESULT> search(const QString &content, int searchType, int searchOpts) override;
    void removeGraph(int idx) override;
    QRectF viewRect(const QModelIndex &subgIdx) override;
    void markDirty() override;
    void clearDirty() override;
    void collaspe(const QModelIndex &subgIdx) override;
    void expand(const QModelIndex &subgIdx) override;
    void setIOProcessing(bool bIOProcessing) override;
    bool IsIOProcessing() const override;
    void beginTransaction(const QString &name) override;
    void endTransaction() override;
    void beginApiLevel() override;
    void endApiLevel() override;
    LinkModel *linkModel(const QModelIndex &subgIdx) const override;
    QModelIndexList findSubgraphNode(const QString &subgName) override;
    int ModelSetData(const QPersistentModelIndex &idx, const QVariant &value, int role,
                             const QString &comment = "") override;
    int undoRedo_updateSubgDesc(const QString &descName, const NODE_DESC &desc) override;
    QModelIndex indexFromPath(const QString &path) override;
    bool addExecuteCommand(QUndoCommand *pCommand) override;
    void setIOVersion(zenoio::ZSG_VERSION ver) override;
    zenoio::ZSG_VERSION ioVersion() const override;
    void setApiRunningEnable(bool bEnable) override;
    bool isApiRunningEnable() const override;
    bool setCustomName(const QModelIndex &subgIdx, const QModelIndex &Idx, const QString &value) override;

    QModelIndex parent(const QModelIndex &) const override;
    int rowCount(const QModelIndex &) const override;
    int columnCount(const QModelIndex &) const override;
    QVariant data(const QModelIndex &, int) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QAbstractItemModel *implModel();
    void onSubgrahSync(const QModelIndex& subgIdx) override;
    void markNodeDataChanged(const QModelIndex&) override;
    void clearNodeDataChanged() override;

public:
    QModelIndex mainIndex() const;
    QList<EdgeInfo> addSubnetNode(
            IGraphsModel *pSubgraphs,
            const QString &subnetName,
            const QString &ident,
            const QString &customName);
    void initSubgraphs(IGraphsModel* pSubgraphs);
    QUndoStack* stack() const;

private:
    IGraphsModel* m_pSubgraphs;
    GraphsTreeModel_impl* m_impl;
    QUndoStack *m_stack;
    QString m_filePath;
    int m_apiLevel;
    bool m_dirty;
    bool m_bIOProcessing;
    bool m_bApiEnableRun;
    zenoio::ZSG_VERSION m_version;
};

#endif