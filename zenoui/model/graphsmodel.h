#ifndef __ZENO_GRAPHS_MODEL_H__
#define __ZENO_GRAPHS_MODEL_H__

#include <QStandardItemModel>
#include <QItemSelectionModel>

class SubGraphModel;

class GraphsModel : public QStandardItemModel
{
	Q_OBJECT
public:
    GraphsModel(QObject* parent = nullptr);
    SubGraphModel* subGraph(const QString& id);
    SubGraphModel *subGraph(int idx);
    int graphCounts() const;

signals:
    void itemSelected(int);

public slots:
    void onCurrentIndexChanged(int);

private slots:
    void onSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

private:
    QItemSelectionModel* m_selection;
    int m_currentIndex;
};

#endif