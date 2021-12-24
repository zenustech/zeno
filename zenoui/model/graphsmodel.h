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
    SubGraphModel* subGraph(const QString& id);
    SubGraphModel *subGraph(int idx);
    SubGraphModel *currentGraph();
    int graphCounts() const;
    NODE_DESCS descriptors() const;
    void setDescriptors(const NODE_DESCS &nodesParams);
    void initDescriptors();
    NODE_DESCS getSubgraphDescs();
    NODE_CATES getCates();

signals:
    void itemSelected(int);

public slots:
    void onCurrentIndexChanged(int);

private slots:
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
    QItemSelectionModel* m_selection;
    NODE_DESCS m_nodesDesc;
    NODE_CATES m_nodesCate;
    int m_currentIndex;
};

#endif