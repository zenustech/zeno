#include "framework.h"
#include "nodesview.h"
#include "styletabwidget.h"
#include "nodescene.h"
#include "layerwidget.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setTabsClosable(true);
    this->tabBar()->setTabsClosable(true);

    connect(this, SIGNAL(tabCloseRequested(int)), this, SIGNAL(tabClosed(int)));
    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(onTabClosed(int)));

    onNewTab();
}

void StyleTabWidget::onTabClosed(int index)
{
    this->removeTab(index);
    //TODO: disconnect
}

void StyleTabWidget::initTabs()
{
    auto pView = new NodesView;
    addTab(pView, "node");
}

NodesView* StyleTabWidget::getCurrentView()
{
    return qobject_cast<NodesView*>(currentWidget());
}

NodesView* StyleTabWidget::getView(int index)
{
    return qobject_cast<NodesView*>(widget(index));
}

QStandardItemModel* StyleTabWidget::getCurrentModel()
{
    NodesView* w = qobject_cast<NodesView*>(this->currentWidget());
    return w->scene()->model();
}

QItemSelectionModel* StyleTabWidget::getSelectionModel()
{
	NodesView* w = qobject_cast<NodesView*>(this->currentWidget());
	return w->scene()->selectionModel();
}

void StyleTabWidget::onNewTab()
{
    auto pView = new NodesView;
    addTab(pView, "node");
    pView->initSkin(":/templates/node-empty.xml");
    pView->initNode();

    emit tabviewActivated(pView->scene()->model());
}
