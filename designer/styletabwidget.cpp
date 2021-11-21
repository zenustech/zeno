#include "framework.h"
#include "styletabwidget.h"
#include "nodescene.h"
#include "nodesview.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setTabsClosable(true);
    this->tabBar()->setTabsClosable(true);

    NodesView* pView = new NodesView;
    addTab(pView, "node");

    connect(this, SIGNAL(tabCloseRequested(int)), this, SIGNAL(tabClosed(int)));
    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(onTabClosed(int)));

    //temp: new tab
    pView->initSkin("E:/zeno/uirender/node-empty.xml");
    pView->initNode();
}

void StyleTabWidget::onTabClosed(int index)
{
    this->removeTab(index);
}
