#include "zenographstabwidget.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <model/graphsmodel.h>
#include <nodesys/zenosubgraphview.h>
#include <comctrl/ziconbutton.h>


ZenoGraphsTabWidget::ZenoGraphsTabWidget(QWidget* parent)
    : QTabWidget(parent)
{
    setAutoFillBackground(false);
}

void ZenoGraphsTabWidget::activate(const QString& subgraphName)
{
    GraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    SubGraphModel* pSubModel = pModel->subGraph(subgraphName);
    int idx = indexOfName(subgraphName);
    if (idx == -1)
    {
        ZenoSubGraphView* pView = new ZenoSubGraphView;
        pView->setModel(pSubModel);
        int idx = addTab(pView, subgraphName);
        setCurrentIndex(idx);
        ZIconButton* pCloseBtn = new ZIconButton(QIcon(":/icons/closebtn.svg"), QSize(14, 14), QColor(), QColor());
        tabBar()->setTabButton(idx, QTabBar::RightSide, pCloseBtn);
        connect(pCloseBtn, &ZIconButton::clicked, this, [=]() {
            this->removeTab(indexOf(pView));
            });
    }
    else
    {
        setCurrentIndex(idx);
    }
}

int ZenoGraphsTabWidget::indexOfName(const QString& subGraphName)
{
    for (int i = 0; i < this->count(); i++)
    {
        if (this->tabText(i) == subGraphName)
        {
            return i;
        }
    }
    return -1;
}

void ZenoGraphsTabWidget::paintEvent(QPaintEvent* e)
{
    _base::paintEvent(e);
}