#ifndef __ZENO_GRAPHS_TABWIDGET_H__
#define __ZENO_GRAPHS_TABWIDGET_H__

#include <QtWidgets>

class ZenoGraphsTabWidget : public QTabWidget
{
    Q_OBJECT
    typedef QTabWidget _base;

public:
    ZenoGraphsTabWidget(QWidget* parent = nullptr);
    void activate(const QString& subgraphName);
    int indexOfName(const QString& subGraphName);

protected:
    void paintEvent(QPaintEvent* e);
};


#endif