#ifndef __STYLE_TABWIDGET_H__
#define __STYLE_TABWIDGET_H__

#include "renderparam.h"
#include "common.h"

class LayerWidget;
class NodesView;

class StyleTabWidget : public QTabWidget
{
    Q_OBJECT
public:
    StyleTabWidget(QWidget* parent = nullptr);
    QStandardItemModel* getCurrentModel();
    QItemSelectionModel* getSelectionModel();
    NodesView* getCurrentView();
    NodesView* getView(int index);

signals:
    void tabClosed(int);
    void tabActivate(NodeParam);
    void tabviewActivated(QStandardItemModel*);

public slots:
    void onNewTab();
    void onTabClosed(int);

private:
    void initTabs();

};

#endif