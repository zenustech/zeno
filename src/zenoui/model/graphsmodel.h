#ifndef __ZENO_GRAPHS_MODEL_H__
#define __ZENO_GRAPHS_MODEL_H__

#include <QStandardItemModel>
#include <QItemSelectionModel>

#include "subgraphmodel.h"
#include "modeldata.h"

class SubGraphModel;

class GraphsModel : public QStandardItemModel
{
	Q_OBJECT
public:
    GraphsModel(QObject* parent = nullptr);
    ~GraphsModel();
    void setFilePath(const QString& fn);
    SubGraphModel* subGraph(const QString& name);
    SubGraphModel *subGraph(int idx);
    SubGraphModel *currentGraph();
    void switchSubGraph(const QString& graphName);
    void newSubgraph(const QString& graphName);
    void reloadSubGraph(const QString& graphName);
    void renameSubGraph(const QString& oldName, const QString& newName);
    QItemSelectionModel* selectionModel() const;
    int graphCounts() const;
    NODE_DESCS descriptors() const;
    void appendSubGraph(SubGraphModel* pGraph);
    void removeGraph(int idx);
    void setDescriptors(const NODE_DESCS &nodesParams);
    void initDescriptors();
    bool isDirty() const;
    void markDirty();
    void clearDirty();
    NODE_DESCS getSubgraphDescs();
    NODE_CATES getCates();
    QString filePath() const;
    QString fileName() const;
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex parent(const QModelIndex& child) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;

signals:
    void graphRenamed(const QString& oldName, const QString& newName);

public slots:
    void onCurrentIndexChanged(int);
    void onRemoveCurrentItem();

private:
    QItemSelectionModel* m_selection;
    NODE_DESCS m_nodesDesc;
    NODE_CATES m_nodesCate;
    QString m_filePath;
    bool m_dirty;
};

#endif