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
    int graphCounts() const;
    NODES_PARAMS getNodesTemplate() const;
    void setNodesTemplate(const NODES_PARAMS& nodesParams);

signals:
    void itemSelected(int);

public slots:
    void onCurrentIndexChanged(int);

private slots:
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
    QItemSelectionModel* m_selection;
    NODES_PARAMS m_nodesDict;
    int m_currentIndex;
};

#endif