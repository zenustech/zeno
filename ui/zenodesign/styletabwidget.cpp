#include "framework.h"
#include "nodesview.h"
#include "styletabwidget.h"
#include "nodescene.h"
#include "layerwidget.h"
#include "nodeswidget.h"


StyleTabWidget::StyleTabWidget(QWidget* parent)
    : QTabWidget(parent)
    , m_currentIndex(0)
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

void StyleTabWidget::addEmptyTab()
{
    auto pTab = new NodesWidget;
    addTab(pTab, "node");
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
    auto pTab = new NodesWidget(this);
    initTab(pTab);
}

void StyleTabWidget::initTab(NodesWidget *pTab)
{
    QString tabName = pTab->fileName();
    //todo: add suffix when repeat name occurs.
    addTab(pTab, tabName);

    int idxTab = indexOf(pTab);
    connect(pTab, &NodesWidget::tabDirtyChanged, [=](bool dirty) {
        QString fileName = pTab->fileName();
        if (fileName.isEmpty())
            fileName = tabName;
        if (dirty) {
            setTabText(idxTab, fileName + "*");
        } else {
            setTabText(idxTab, fileName);
        }
    });
    setCurrentWidget(pTab);
    emit tabviewActivated(pTab->model());
}

void StyleTabWidget::openFile(const QString &filePath)
{
    auto pTab = new NodesWidget(filePath, this);
    initTab(pTab);
}
