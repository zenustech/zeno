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
    SubGraphModel* subGraph(const QString& id);
    SubGraphModel *subGraph(int idx);
    SubGraphModel *currentGraph();
    void switchSubGraph(const QString& graphName);
    void newSubgraph(const QString& graphName);
    void reloadSubGraph(const QString& graphName);
    QItemSelectionModel* selectionModel() const;
    int graphCounts() const;
    NODE_DESCS descriptors() const;
    void appendSubGraph(SubGraphModel* pGraph);
    void removeGraph(int idx);
    void setDescriptors(const NODE_DESCS &nodesParams);
    void initDescriptors();
    NODE_DESCS getSubgraphDescs();
    NODE_CATES getCates();

public slots:
    void onCurrentIndexChanged(int);
    void onRemoveCurrentItem();

private:
    QItemSelectionModel* m_selection;
    NODE_DESCS m_nodesDesc;
    NODE_CATES m_nodesCate;
    QString m_filePath;
};

#endif