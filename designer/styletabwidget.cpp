#include "framework.h"
#include "styletabwidget.h"
#include "nodescene.h"
#include "nodesview.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setTabsClosable(true);
    this->tabBar()->setTabsClosable(true);
    addTab(new NodesView, "node");
    addTab(new QWidget, "Node2");

    connect(this, SIGNAL(tabCloseRequested(int)), this, SIGNAL(tabClosed(int)));
    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(onTabClosed(int)));
}

void StyleTabWidget::onTabClosed(int index)
{
    this->removeTab(index);
}
