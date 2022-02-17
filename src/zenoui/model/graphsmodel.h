#ifndef __ZENO_GRAPHS_MODEL_H__
#define __ZENO_GRAPHS_MODEL_H__

#include <QStandardItemModel>
#include <QItemSelectionModel>

#include <zenoui/include/igraphsmodel.h>

#include "subgraphmodel.h"
#include "modeldata.h"

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
public:
    GraphsModel(QObject* parent = nullptr);
    ~GraphsModel();
    void setFilePath(const QString& fn);
    SubGraphModel* subGraph(const QString& name) const;
    SubGraphModel *subGraph(int idx) const;
    SubGraphModel *currentGraph();
    void switchSubGraph(const QString& graphName);
    void newSubgraph(const QString& graphName);
    void reloadSubGraph(const QString& graphName);
    void renameSubGraph(const QString& oldName, const QString& newName);
    QItemSelectionModel* selectionModel() const;
    NODE_DESCS descriptors() const override;
    void setDescriptors(const NODE_DESCS& nodesParams) override;
    void appendSubGraph(SubGraphModel* pGraph);
    void removeGraph(int idx) override;
    void initDescriptors() override;
    bool isDirty() const override;
    void markDirty();
    void clearDirty();
    NODE_DESCS getSubgraphDescs();
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

    //IGraphsModel
	void beginTransaction(const QModelIndex& subGpIdx) override;
	void endTransaction(const QModelIndex& subGpIdx) override;
	QModelIndex index(const QString& id, const QModelIndex& subGpIdx) override;
    QModelIndex index(int r, const QModelIndex& subGpIdx) override;
	QVariant data2(const QModelIndex& subGpIdx, const QModelIndex& index, int role) override;
	void setData2(const QModelIndex& subGpIdx, const QModelIndex& index, const QVariant& value, int role) override;
    int itemCount(const QModelIndex& subGpIdx) const override;
	void addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
    void insertRow(int row, const NODE_DATA& nodeData, const QModelIndex& subGpIdx) override;
	void appendNodes(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx) override;
	void removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction = false) override;
	void removeNode(int row, const QModelIndex& subGpIdx) override;
    void removeLinks(const QList<QPersistentModelIndex>& info, const QModelIndex& subGpIdx) override;
    void removeLink(const QPersistentModelIndex& linkIdx, const QModelIndex& subGpIdx) override;
	void removeSubGraph(const QString& name) override;
	void addLink(const EdgeInfo& info, const QModelIndex& subGpIdx) override;
	void updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx) override;
	void updateSocket(const QString& id, SOCKET_UPDATE_INFO info, const QModelIndex& subGpIdx) override;
	NODE_DATA itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const override;
	QString name(const QModelIndex& subGpIdx) const override;
	void setName(const QString& name, const QModelIndex& subGpIdx) override;
	void replaceSubGraphNode(const QString& oldName, const QString& newName, const QModelIndex& subGpIdx) override;
	NODES_DATA nodes(const QModelIndex& subGpIdx) override;
	void clear(const QModelIndex& subGpIdx) override;
	void reload(const QModelIndex& subGpIdx) override;
	void onModelInited();
	void undo() override;
	void redo() override;
    QModelIndexList searchInSubgraph(const QString& objName, const QModelIndex& subgIdx) override;
    QStandardItemModel* linkModel() const;
    QModelIndex getSubgraphIndex(const QModelIndex& linkIdx);

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
    QVector<SubGraphModel*> m_subGraphs;
    QItemSelectionModel* m_selection;
    QStandardItemModel* m_linkModel;
    NODE_DESCS m_nodesDesc;
    NODE_CATES m_nodesCate;
    QString m_filePath;
    QUndoStack* m_stack;
    bool m_dirty;
};

#endif