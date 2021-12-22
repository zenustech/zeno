#include "zenosubgraphscene.h"
#include "../model/graphsmodel.h"
#include "../model/subgraphmodel.h"
#include "../model/modelrole.h"
#include "zenosubgraphview.h"
#include "zenographswidget.h"
#include "../util/uihelper.h"


ZenoGraphsWidget::ZenoGraphsWidget(QWidget* parent)
    : QStackedWidget(parent)
    , m_model(nullptr)
{
}

void ZenoGraphsWidget::setGraphsModel(GraphsModel* pModel)
{
    if (!pModel)
        return;

    m_model = pModel;
    connect(m_model, SIGNAL(rowsRemoved(const QModelIndex&, int, int)), this, SLOT(onRowsRemoved(const QModelIndex&, int, int)));
    connect(m_model, SIGNAL(itemSelected(int)), this, SLOT(setCurrentIndex(int)));
    //connect: delete btn -> model remove row.
    for (auto it = m_views.begin(); it != m_views.end(); ++it) {
        removeWidget(it->second);
    }
    m_views.clear();

    for (int i = 0; i < m_model->graphCounts(); i++)
    {
        SubGraphModel* pSubModel = m_model->subGraph(i);
        Q_ASSERT(pSubModel);
        ZenoSubGraphView* pView = new ZenoSubGraphView;
        pView->setModel(pSubModel);
        addWidget(pView);
        m_views.insert(std::pair(pSubModel->name(), pView));
    }
}

void ZenoGraphsWidget::initDescriptors()
{
    if (m_descs.isEmpty())
    {
        m_descs = UiHelper::loadDescsFromTempFile();
    }
}

void ZenoGraphsWidget::onSwitchGraph(const QString& graphName)
{
    auto iter = m_views.find(graphName);
    Q_ASSERT(iter != m_views.end());
    setCurrentWidget(iter->second);
}

void ZenoGraphsWidget::onRowsRemoved(const QModelIndex &parent, int first, int last)
{
}