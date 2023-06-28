#ifndef __GRAPHICS_IMPL_TREEMODEL_H__
#define __GRAPHICS_IMPL_TREEMODEL_H__

#include <QtWidgets>
#include "modeldata.h"
#include "common.h"

class LinkModel;
class TreeNodeItem;
class GraphsTreeModel;
class IGraphsModel;
class SubGraphModel;

class GraphsTreeModel_impl : public QStandardItemModel
{
    Q_OBJECT
public:
    GraphsTreeModel_impl(GraphsTreeModel* pModel, QObject *parent = nullptr);
    ~GraphsTreeModel_impl();

    //impl for IGraphsModel
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex index(const QString &subGraphName) const;
    QModelIndex index(const QString &id, const QModelIndex &subGpIdx);
    QModelIndex index(int r, const QModelIndex &subGpIdx);
    QModelIndex nodeIndex(const QString &ident);
    QModelIndex mainIndex() const;
    /* end: node index: */

    QModelIndex nodeIndex(uint32_t sid, uint32_t nodeid);
    QModelIndex subgIndex(uint32_t sid);

    int itemCount(const QModelIndex &subGpIdx) const;

    QModelIndex linkIndex(const QModelIndex &subgIdx, int r);
    QModelIndex linkIndex(const QModelIndex &subgIdx, const QString &outNode, const QString &outSock,
                          const QString &inNode, const QString &inSock);

    void addNode(const NODE_DATA &nodeData, const QModelIndex &subGpIdx, bool enableTransaction = false);
    QList<EdgeInfo> addSubnetNode(
            IGraphsModel *pSubgraphs,
            const QString &subnetName,
            const QString &ident,
            const QString &customName);

    void setNodeData(const QModelIndex &nodeIndex, const QModelIndex &subGpIdx, const QVariant &value,
                     int role);
    void importNodes(const QMap<QString, NODE_DATA> &nodes, const QList<EdgeInfo> &links, const QPointF &pos,
                     const QModelIndex &subGpIdx, bool enableTransaction = false);
    void removeNode(const QString &nodeid, const QModelIndex &subGpIdx, bool enableTransaction = false);

    QModelIndex addLink(const QModelIndex &subgIdx, const QModelIndex &fromSock, const QModelIndex &toSock,
                        bool enableTransaction = false);
    QModelIndex addLink(const EdgeInfo &info, bool enableTransaction = false);
    void removeLink(const QModelIndex &linkIdx, bool enableTransaction = false);
    void removeLink(const QModelIndex &subgIdx, const EdgeInfo &linkIdx, bool enableTransaction = false);
    bool IsSubGraphNode(const QModelIndex &nodeIdx) const;

    QModelIndex fork(const QModelIndex &subgIdx, const QModelIndex &subnetNodeIdx);

    void updateParamInfo(const QString &id, PARAM_UPDATE_INFO info, const QModelIndex &subGpIdx,
                         bool enableTransaction = false);
    void updateSocketDefl(const QString &id, PARAM_UPDATE_INFO info, const QModelIndex &subGpIdx,
                          bool enableTransaction = false);
    void updateNodeStatus(const QString &nodeid, STATUS_UPDATE_INFO info, const QModelIndex &subgIdx,
                          bool enableTransaction = false);
    void updateBlackboard(const QString &id, const QVariant &blackboard, const QModelIndex &subgIdx,
                          bool enableTransaction);

    NODE_DATA itemData(const QModelIndex &index, const QModelIndex &subGpIdx) const;
    void exportSubgraph(const QModelIndex& subGpIdx, NODES_DATA& nodes, LINKS_DATA& links) const;
    void setName(const QString &name, const QModelIndex &subGpIdx);

    QModelIndexList searchInSubgraph(const QString &objName, const QModelIndex &subgIdx);
    QList<SEARCH_RESULT> search(const QString &content, int searchType, int searchOpts);
    QRectF viewRect(const QModelIndex &subgIdx);
    void collaspe(const QModelIndex &subgIdx);
    void expand(const QModelIndex &subgIdx);
    LinkModel *linkModel(const QModelIndex &subgIdx) const;
    int ModelSetData(const QPersistentModelIndex &idx, const QVariant &value, int role,
                     const QString &comment = "");
    QModelIndex indexFromPath(const QString &path);
    GraphsTreeModel* model() const;
    bool setCustomName(const QModelIndex &subgIdx, const QModelIndex &Idx, const QString &value);
    void initMainGraph();
    void clear();
    void renameSubGraph(const QString &oldName, const QString &newName);
    void appendSubGraphNode(TreeNodeItem *pSubgraph);
    void removeSubGraphNode(TreeNodeItem *pSubgraph);
    void onSubgrahSync(const QModelIndex& subgIdx);
    QModelIndex extractSubGraph(const QModelIndexList& nodes, const QModelIndexList& links,
        const QModelIndex& fromSubg, const QString& toSubg, bool enableTrans = false);

private:
    void onSubIOAddRemove(TreeNodeItem* pSubgraph, const QModelIndex& addedNodeIdx, bool bInput, bool bInsert);
    bool onSubIOAdd(TreeNodeItem* pSubgraph, NODE_DATA nodeData);
    bool onListDictAdd(TreeNodeItem* pSubgraph, NODE_DATA nodeData);
    TreeNodeItem* _fork(const QString& currentPath,
                        IGraphsModel *pSubgraphs,
                        const QString &subnetName,
                        const NODE_DATA &nodeData,
                        QList<EdgeInfo>& newLinks);
    QList<SEARCH_RESULT> search_impl(
            const QModelIndex& root,
            const QString &content,
            int searchType,
            int searchOpts,
            bool bRecursivly);
    bool search_result(const QModelIndex& root, const QModelIndex& index, const QString& content, int searchType, int searchOpts, QList<SEARCH_RESULT>& results);

    TreeNodeItem* m_main;       //not invisible root.
    LinkModel* m_linkModel;
    GraphsTreeModel* m_pModel;
    QMap<QString, QList<TreeNodeItem *>> m_treeNodeItems;//key: node name
};

#endif