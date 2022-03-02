#ifndef __ZENO_GRAPHS_MODEL_H__
#define __ZENO_GRAPHS_MODEL_H__

#include <QStandardItemModel>
#include <QItemSelectionModel>

#include <zenoui/include/igraphsmodel.h>
#include "../nodesys/zenosubgraphscene.h"
#include "subgraphmodel.h"
#include <zenoui/model/modeldata.h>

class SubGraphModel;

/*
GraphsModel is a "plain" model, which contains subgraphModel for each subgraph.
Modification at any SubgraphModel will apply to all subnet nodes associated to this subgraph.
the implemenation is compatible with the graphs organization in zeno1.0.
*/

class GraphsModel : public IGraphsModel
{
    Q_OBJECT
    typedef IGraphsModel _base;

    struct SUBMODEL_SCENE
    {
        SubGraphModel* pModel;
        ZenoSubGraphScene* pScene;
        SUBMODEL_SCENE() : pModel(nullptr), pScene(nullptr) {}
    };

public:
    GraphsModel(QObject* parent = nullptr);
    ~GraphsModel();
    void setFilePath(const QString& fn);
    SubGraphModel* subGraph(const QString& name) const;
    SubGraphModel *subGraph(int idx) const;
    SUBMODEL_SCENE subGraphScene(int idx) const;
    SubGraphModel *currentGraph();
    void switchSubGraph(const QString& graphName);
    void newSubgraph(const QString& graphName);
    void reloadSubGraph(const QString& graphName);
    void renameSubGraph(const QString& oldName, const QString& newName);
    QItemSelectionModel* selectionModel() const;
    NODE_DESCS descriptors() const override;
    void setDescriptors(const NODE_DESCS& nodesParams) override;
    void appendSubGraph(SubGraphModel* pGraph);
    QModelIndex fork(const QModelIndex& subgIdx, const QModelIndex& subnetNodeIdx) override;
    void removeGraph(int idx) override;
    void initDescriptors() override;
    bool isDirty() const override;
    void markDirty();
    void clearDirty();
    NODE_CATES getCates() override;
    QString filePath() const;
    QString fileName() const override;
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex index(const QString& subGraphName) const;
    QModelIndex indexBySubModel(SubGraphModel* pSubModel) const;
    QModelIndex linkIndex(int r);
    QModelIndex parent(const QModelIndex& child) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const;
    void revert(const QModelIndex& idx);

    //IGraphsModel
	void beginTransaction(const QString& name) override;
	void endTransaction() override;
	QModelIndex index(const QString& id, const QModelIndex& subGpIdx) override;
    QModelIndex index(int r, const QModelIndex& subGpIdx) override;
	QVariant data2(const QModelIndex& subGpIdx, const QModelIndex& index, int role) override;
	void setData2(const QModelIndex& subGpIdx, const QModelIndex& index, const QVariant& value, int role) override;
    int itemCount(const QModelIndex& subGpIdx) const override;
	void addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void insertRow(int row, const NODE_DATA& nodeData, const QModelIndex& subGpIdx) override;
	void appendNodes(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void importNodeLinks(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx) override;
	void removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
	void removeNode(int row, const QModelIndex& subGpIdx) override;
    void removeLinks(const QList<QPersistentModelIndex>& info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void removeLink(const QPersistentModelIndex& linkIdx, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
	void removeSubGraph(const QString& name) override;
    QModelIndex addLink(const EdgeInfo& info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void updateLinkInfo(const QPersistentModelIndex& linkIdx, const LINK_UPDATE_INFO& info, bool enableTransaction = false) override;
    QVariant getParamValue(const QString& id, const QString& name, const QModelIndex& subGpIdx) override;
	void updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void updateSubnetIO(QPersistentModelIndex subgIdx, const QString& subNodeId, const SOCKET_UPDATE_INFO& info) override;
	void updateSocket(const QString& id, SOCKET_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx) override;
    QVariant getNodeStatus(const QString& id, int role, const QModelIndex& subGpIdx) override;
    void updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction = false) override;
	NODE_DATA itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const override;
	QString name(const QModelIndex& subGpIdx) const override;
	void setName(const QString& name, const QModelIndex& subGpIdx) override;
	void replaceSubGraphNode(const QString& oldName, const QString& newName, const QModelIndex& subGpIdx) override;
	NODES_DATA nodes(const QModelIndex& subGpIdx) override;
	void clearSubGraph(const QModelIndex& subGpIdx) override;
    void clear() override;
	void reload(const QModelIndex& subGpIdx) override;
	void onModelInited();
	void undo() override;
	void redo() override;
    QModelIndexList searchInSubgraph(const QString& objName, const QModelIndex& subgIdx) override;
    QStandardItemModel* linkModel() const;
    QModelIndex getSubgraphIndex(const QModelIndex& linkIdx);
    QGraphicsScene* scene(const QModelIndex& subgIdx) override;
    QRectF viewRect(const QModelIndex& subgIdx) override;

signals:
    void graphRenamed(const QString& oldName, const QString& newName);

public slots:
    void onCurrentIndexChanged(int);
    void onRemoveCurrentItem();

    void on_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
    void on_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
    void on_rowsInserted(const QModelIndex& parent, int first, int last);
    void on_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void on_rowsRemoved(const QModelIndex& parent, int first, int last);

    void on_linkDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
	void on_linkAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void on_linkInserted(const QModelIndex& parent, int first, int last);
	void on_linkAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_linkRemoved(const QModelIndex& parent, int first, int last);

private:
    NODE_DESCS getSubgraphDescs();
    NODE_DESCS getCoreDescs();

    QVector<SUBMODEL_SCENE> m_subGraphs;
    QItemSelectionModel* m_selection;
    QStandardItemModel* m_linkModel;
    NODE_DESCS m_nodesDesc;
    NODE_CATES m_nodesCate;
    QString m_filePath;
    QUndoStack* m_stack;
    bool m_dirty;
};

#endif
