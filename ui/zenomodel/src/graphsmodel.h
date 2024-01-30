#ifndef __ZENO_GRAPHS_MODEL_H__
#define __ZENO_GRAPHS_MODEL_H__

#include <QStandardItemModel>
#include <QItemSelectionModel>

#include <zenomodel/include/igraphsmodel.h>
#include "subgraphmodel.h"
#include "linkmodel.h"
#include "modeldata.h"
#include <stack>

class SubGraphModel;
class GraphsModel;
class ApiLevelScope;

/*
GraphsModel is a "plain" model, which contains subgraphModel for each subgraph.
Modification at any SubgraphModel will apply to all subnet nodes associated to this subgraph.
the implemenation is compatible with the graphs organization in zeno1.0.
*/

class GraphsModel : public IGraphsModel
{
    Q_OBJECT
    typedef IGraphsModel _base;

public:
    GraphsModel(QObject* parent = nullptr);
    ~GraphsModel();
    SubGraphModel* subGraph(const QString& name) const;
    SubGraphModel *subGraph(int idx) const;
    SubGraphModel *currentGraph();
    void switchSubGraph(const QString& graphName) override;
    void newSubgraph(const QString& graphName, SUBGRAPH_TYPE type = SUBGRAPH_TYPE::SUBGRAPH_NOR) override;
    bool newMaterialSubgraph(const QModelIndex& subgIdx, const QString& graphName, const QPointF& pos) override;
    void initMainGraph() override;
    void renameSubGraph(const QString& oldName, const QString& newName) override;
    QItemSelectionModel* selectionModel() const;
    NODE_DESCS descriptors() const override;
    bool appendSubnetDescsFromZsg(const QList<NODE_DESC>& descs, bool bImport = false) override;
    bool getDescriptor(const QString& descName, NODE_DESC& desc) override;
    bool updateSubgDesc(const QString& descName, const NODE_DESC& desc) override;
    //NODE_DESC
    void appendSubGraph(SubGraphModel* pGraph);
    QModelIndex fork(const QModelIndex& subgIdx, const QModelIndex& subnetNodeIdx) override;
    QModelIndex forkMaterial(const QModelIndex& subgIdx, const QModelIndex& subnetNodeIdx, const QString&subgName, const QString& mtlid, const QString& mtlid_old) override;
    void removeGraph(int idx) override;
    bool isDirty() const override;
    void markDirty() override;
    void clearDirty() override;
    NODE_CATES getCates() override;
    QString filePath() const override;
    QString fileName() const override;
    void setFilePath(const QString& fn) override;
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex nodeIndex(uint32_t sid, uint32_t nodeid) override;
    QModelIndex subgIndex(uint32_t sid) override;
    QModelIndex paramIndex(const QModelIndex& subgIdx, const QModelIndex& nodeIdx, const QString& name, bool bInput) override;
    QModelIndex index(const QString& subGraphName) const override;
    QModelIndex indexBySubModel(SubGraphModel* pSubModel) const;
    QModelIndex indexFromPath(const QString& path) override;
    QModelIndex indexFromPath(const QStringList& lst) override;
    QModelIndex linkIndex(const QModelIndex& subgIdx, int r) override;
    QModelIndex linkIndex(const QModelIndex& subgIdx, const QString& outNode, const QString& outSock, const QString& inNode, const QString& inSock) override;

    QModelIndex parent(const QModelIndex& child) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    void revert(const QModelIndex& idx);

    //IGraphsModel

    QModelIndex index(const QString& id, const QModelIndex& subGpIdx) override;
    QModelIndex nodeIndex(const QString &ident) override;
    QModelIndex index(int r, const QModelIndex& subGpIdx) override;
    int itemCount(const QModelIndex& subGpIdx) const override;
    void addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void appendNodes(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx, bool enableTransaction = false);
    void setNodeData(const QModelIndex& nodeIndex, const QModelIndex& subGpIdx, const QVariant& value, int role) override;
    void importNodes(
            const QMap<QString, NODE_DATA>& nodes,
            const QList<EdgeInfo>& links,
            const QPointF& pos,
            const QModelIndex& subGpIdx,
            bool enableTransaction = false) override;
    void removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void removeNode(int row, const QModelIndex& subGpIdx);
    void removeLink(const QModelIndex& linkIdx, bool enableTransaction = false) override;
    void removeLink(const QModelIndex& subgIdx, const EdgeInfo& linkIdx, bool enableTransaction = false) override;
    void removeLegacyLink(const QModelIndex& linkIdx) override;
    void removeSubGraph(const QString& name) override;
    QModelIndex addLink(const QModelIndex& subgIdx, const QModelIndex& fromSock, const QModelIndex& toSock, bool enableTransaction = false) override;
    QModelIndex addLink(const QModelIndex& subgIdx, const EdgeInfo& info, bool enableTransaction = false) override;
    void addLegacyLink(const QModelIndex& subgIdx, const QModelIndex& fromSock, const QModelIndex& toSock);

    void updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction = false) override;
    void updateBlackboard(const QString &id, const QVariant &blackboard, const QModelIndex &subgIdx,
                          bool enableTransaction) override;

    QModelIndex extractSubGraph(const QModelIndexList& nodes, const QModelIndexList& links, const QModelIndex& fromSubg, const QString& toSubg, bool enableTrans = false) override;
    bool IsSubGraphNode(const QModelIndex& nodeIdx) const override;
    bool IsDeprecatedhNode(const QModelIndex& nodeIdx) override;

	NODE_DATA itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const override;
	void setName(const QString& name, const QModelIndex& subGpIdx) override;
	void clearSubGraph(const QModelIndex& subGpIdx) override;
    void clear() override;
	void undo() override;
	void redo() override;
    QModelIndexList searchInSubgraph(const QString& objName, const QModelIndex& subgIdx) override;
    QModelIndexList subgraphsIndice() const override;
    QModelIndexList subgraphsIndice(SUBGRAPH_TYPE type) const;
    LinkModel* linkModel(const QModelIndex& subgIdx) const override;
    LinkModel* legacyLinks(const QModelIndex& subgIdx) const override;
    QModelIndex getSubgraphIndex(const QModelIndex& linkIdx);
    QRectF viewRect(const QModelIndex& subgIdx) override;
    QList<SEARCH_RESULT> search(
                        const QString &content,
                        int searchType,
                        int searchOpts,
                        QVector<SubGraphModel*> vec = QVector<SubGraphModel *>()) const override;
	void collaspe(const QModelIndex& subgIdx) override;
	void expand(const QModelIndex& subgIdx) override;

    bool hasDescriptor(const QString& nodeName) const;
    void beginTransaction(const QString& name) override;
	void endTransaction() override;
    void beginApiLevel() override;
    void endApiLevel() override;
    void setIOProcessing(bool bIOProcessing) override;
    bool IsIOProcessing() const override;
    QModelIndexList findSubgraphNode(const QString& subgName) override;
    int ModelSetData(
        const QPersistentModelIndex& idx,
        const QVariant& value,
        int role,
        const QString& comment = "") override;
    int undoRedo_updateSubgDesc(const QString &descName, const NODE_DESC &desc) override;
    bool addExecuteCommand(QUndoCommand* pCommand) override; 
    void setIOVersion(zenoio::ZSG_VERSION ver) override;
    zenoio::ZSG_VERSION ioVersion() const override;
    void setApiRunningEnable(bool bEnable) override;
    bool isApiRunningEnable() const override;
    bool setCustomName(const QModelIndex &subgIdx, const QModelIndex &Idx, const QString &value) const override;
    void markNodeDataChanged(const QModelIndex& idx, bool recursively = true) override;
    void markNodeDataUnchanged(const QModelIndex& idx) override;
    void markNotDescNode() override;
    bool hasNotDescNode() const override;
    void clearNodeDataChanged() override;
    QStringList subgraphsName() const override;

    void addNetLabel(const QModelIndex& subgIdx, const QModelIndex& sock, const QString& name) override;
    void removeNetLabel(const QModelIndex& subgIdx, const QModelIndex& trigger) override;
    void updateNetLabel(const QModelIndex& subgIdx, const QModelIndex& trigger, const QString& oldName, const QString& newName, bool enableTransaction = false) override;

    bool addCommandParam(const QString& path, const CommandParam& val) override;
    void removeCommandParam(const QString& path) override;
    bool updateCommandParam(const QString& path, const CommandParam& newVal) override;
    FuckQMap<QString, CommandParam> commandParams() const override;

    QModelIndex getNetOutput(const QModelIndex& subgIdx, const QString& name) const override;
    QList<QModelIndex> getNetInputs(const QModelIndex& subgIdx, const QString& name) const override;
    QStringList dumpLabels(const QModelIndex& subgIdx) const override;
    void addNetLabel_impl(const QModelIndex& subgIdx, const QModelIndex& sock, const QString& name, bool enableTransaction = false);
    void removeNetLabel_impl(const QModelIndex& subgIdx, const QModelIndex& trigger, const QString& name, bool enableTransaction = false);

signals:
    void graphRenamed(const QString& oldName, const QString& newName);

public slots:
    void onCurrentIndexChanged(int);
    void onRemoveCurrentItem();

    void on_subg_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
    void on_subg_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
    void on_subg_rowsInserted(const QModelIndex& parent, int first, int last);
    void on_subg_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void on_subg_rowsRemoved(const QModelIndex& parent, int first, int last);

    void on_linkDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
	void on_linkAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void on_linkInserted(const QModelIndex& parent, int first, int last);
	void on_linkAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_linkRemoved(const QModelIndex& parent, int first, int last);

private:
    NODE_DESCS getCoreDescs();
    void _markNodeChanged(const QModelIndex& idx, bool recursively = true);
    void _markSubnodesChange(SubGraphModel* pSubg);
    void _findReference(
        const QString& subgraphName,
        QModelIndexList& refNodesInMain //the reference can be directly or indirectly nodes.
    );

    void parseDescStr(const QString& descStr, QString& name, QString& type, QVariant& defl);
    void onSubIOAddRemove(SubGraphModel* pSubModel, const QModelIndex& idx, bool bInput, bool bInsert);
    bool onSubIOAdd(SubGraphModel* pGraph, NODE_DATA nodeData2);
    bool onListDictAdd(SubGraphModel* pGraph, NODE_DATA nodeData2);

    QModelIndex _createIndex(SubGraphModel* pSubModel) const;
    void initDescriptors();
    NODE_DESC getSubgraphDesc(SubGraphModel* pModel);
    void registerCate(const NODE_DESC& desc);
    NODE_DATA _fork(const QString& forkSubgName);
    QString uniqueSubgraph(QString orginName);

    void onApiBatchFinished();

    QHash<QString, SubGraphModel*> m_subGraphs;
    QHash<QString, int> m_key2Row;
    QHash<int, QString> m_row2Key;

    QHash<uint32_t, QString> m_id2name;
    QHash<QString, uint32_t> m_name2id;
    QItemSelectionModel* m_selection;

    //LinkModel* m_linkModel;
    QHash<QString, LinkModel*> m_linksGroup;
    QHash<QString, LinkModel*> m_legacyLinks;
    QSet<QPersistentModelIndex> m_changedNodes;

    FuckQMap<QString, CommandParam> m_commandParams;//key:path  value:name

    NODE_DESCS m_nodesDesc;
    NODE_DESCS m_subgsDesc;
    NODE_CATES m_nodesCate;
    QString m_filePath;
    QUndoStack* m_stack;
    std::stack<bool> m_retStack;
    int m_apiLevel;
    bool m_dirty;
    bool m_bIOProcessing;
    bool m_bApiEnableRun;
    bool m_bHasNotDesc;         //has nodes which are not descripied by core decs.
    zenoio::ZSG_VERSION m_version;

    friend class ApiLevelScope;
};

#endif
