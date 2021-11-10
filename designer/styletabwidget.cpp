#include "framework.h"
#include "styletabwidget.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setTabsClosable(true);
    this->tabBar()->setTabsClosable(true);
    addTab(new QWidget, "Node1");
    addTab(new QWidget, "Node2");

    connect(this, SIGNAL(tabCloseRequested(int)), this, SIGNAL(tabClosed(int)));
    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(onTabClosed(int)));
}

void StyleTabWidget::onTabClosed(int index)
{
    this->removeTab(index);
}
