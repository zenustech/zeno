#include "framework.h"
#include "styletabwidget.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setTabsClosable(true);
    this->tabBar()->setTabsClosable(true);
    addTab(new QWidget, "Node1");
    addTab(new QWidget, "Node2");
}