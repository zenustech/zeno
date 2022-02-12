#ifndef __ZENO_GRAPHS_TABWIDGET_H__
#define __ZENO_GRAPHS_TABWIDGET_H__

#include <QtWidgets>

#include <zenoui/include/igraphsmodel.h>

class IGraphsModel;

class ZenoGraphsTabWidget : public QTabWidget
{
    Q_OBJECT
    typedef QTabWidget _base;

public:
    ZenoGraphsTabWidget(QWidget* parent = nullptr);
    void activate(const QString& subgraphName);
    int indexOfName(const QString& subGraphName);
    void resetModel(IGraphsModel* pModel);

public slots:
    void onSubGraphsToRemove(const QModelIndex&, int, int);
    void onModelReset();
    void onSubGraphRename(const QString& oldName, const QString& newName);

protected:
    void paintEvent(QPaintEvent* e);

private:
    IGraphsModel* m_model;
};


#endif