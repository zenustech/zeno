#include "zenographstabwidget.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <model/graphsmodel.h>
#include <nodesys/zenosubgraphview.h>
#include <comctrl/ziconbutton.h>


ZenoGraphsTabWidget::ZenoGraphsTabWidget(QWidget* parent)
    : QTabWidget(parent)
    , m_model(nullptr)
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

void ZenoGraphsTabWidget::resetModel(GraphsModel* pModel)
{
    m_model = pModel;
    connect(pModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)), this, SLOT(onSubGraphsToRemove(const QModelIndex&, int, int)));
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
    connect(pModel, SIGNAL(graphRenamed(const QString&, const QString&)), this, SLOT(onSubGraphRename(const QString&, const QString&)));
}

void ZenoGraphsTabWidget::onSubGraphsToRemove(const QModelIndex& parent, int first, int last)
{
    QTabBar* pTabBar = tabBar();
    for (int r = first; r <= last; r++)
    {
        SubGraphModel* pSubModel = m_model->subGraph(r);
        const QString& name = pSubModel->name();
        pTabBar->removeTab(indexOfName(name));
    }
}

void ZenoGraphsTabWidget::onModelReset()
{
    clear();
    m_model = nullptr;
}

void ZenoGraphsTabWidget::onSubGraphRename(const QString& oldName, const QString& newName)
{
    int idx = indexOfName(oldName);
    Q_ASSERT(idx != -1);
    QTabBar* pTabBar = tabBar();
    pTabBar->setTabText(idx, newName);
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