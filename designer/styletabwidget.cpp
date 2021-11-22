#include "framework.h"
#include "styletabwidget.h"
#include "nodescene.h"
#include "nodesview.h"
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
    pView->initSkin("C:/editor/uirender/node-empty.xml");
    pView->initNode();

    emit tabviewActivated(pView->scene()->model());

 //   connect(pView->scene(), SIGNAL(imageElemOperated(ImageElement, NODE_ID)),
 //       this, SIGNAL(imageElemOperated(ImageElement, NODE_ID)));
	//connect(pView->scene(), SIGNAL(textElemOperated(ImageElement, NODE_ID)),
	//	this, SIGNAL(textElemOperated(ImageElement, NODE_ID)));
	//connect(pView->scene(), SIGNAL(compElementOperated(ImageElement, NODE_ID)),
	//	this, SIGNAL(compElementOperated(ImageElement, NODE_ID)));
}
