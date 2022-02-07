#ifndef __ZENO_GRAPHS_TABWIDGET_H__
#define __ZENO_GRAPHS_TABWIDGET_H__

#include <QtWidgets>

class GraphsModel;

class ZenoGraphsTabWidget : public QTabWidget
{
    Q_OBJECT
    typedef QTabWidget _base;

public:
    ZenoGraphsTabWidget(QWidget* parent = nullptr);
    void activate(const QString& subgraphName);
    int indexOfName(const QString& subGraphName);
    void resetModel(GraphsModel* pModel);

public slots:
    void onSubGraphsToRemove(const QModelIndex&, int, int);
    void onModelReset();
    void onSubGraphRename(const QString& oldName, const QString& newName);

protected:
    void paintEvent(QPaintEvent* e);

private:
    GraphsModel* m_model;
};


#endif