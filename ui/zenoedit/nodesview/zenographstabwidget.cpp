#include "zenographstabwidget.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include "../nodesys/zenosubgraphview.h"
#include <comctrl/ziconbutton.h>


ZenoGraphsTabWidget::ZenoGraphsTabWidget(QWidget* parent)
    : QTabWidget(parent)
    , m_model(nullptr)
{
    setAutoFillBackground(false);
}

void ZenoGraphsTabWidget::activate(const QString& subgraphName)
{
    auto graphsMgm = zenoApp->graphsManagment();
    IGraphsModel* pModel = graphsMgm->currentModel();
    int idx = indexOfName(subgraphName);
    if (idx == -1)
    {
        const QModelIndex& subgIdx = pModel->index(subgraphName);
		ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(pModel->scene(subgIdx));
		Q_ASSERT(pScene);

        ZenoSubGraphView* pView = new ZenoSubGraphView;
        pView->initScene(pScene);
        int idx = addTab(pView, subgraphName);
        setCurrentIndex(idx);
        ZIconButton* pCloseBtn = new ZIconButton(QIcon(":/icons/closebtn.svg"), QSize(14, 14), QColor(), QColor());
        tabBar()->setTabButton(idx, QTabBar::RightSide, pCloseBtn);
        connect(pCloseBtn, &ZIconButton::clicked, this, [=]() {
            removeTab(indexOf(pView));
        });
    }
    else
    {
        setCurrentIndex(idx);
    }
}

void ZenoGraphsTabWidget::resetModel(IGraphsModel* pModel)
{
    m_model = pModel;
    connect(pModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)), this, SLOT(onSubGraphsToRemove(const QModelIndex&, int, int)));
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
    connect(pModel, SIGNAL(graphRenamed(const QString&, const QString&)), this, SLOT(onSubGraphRename(const QString&, const QString&)));
}

void ZenoGraphsTabWidget::onSubGraphsToRemove(const QModelIndex& parent, int first, int last)
{
    for (int r = first; r <= last; r++)
    {
        QModelIndex subgIdx = m_model->index(r, 0);
        const QString& name = m_model->name(subgIdx);
        int idx = indexOfName(name);
        removeTab(idx);
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
    if (idx != -1)
    {
		QTabBar* pTabBar = tabBar();
		pTabBar->setTabText(idx, newName);
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