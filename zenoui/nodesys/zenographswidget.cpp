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

    do {
        removeWidget(widget(0));
    } while (count() > 0);

    for (int i = 0; i < pModel->graphCounts(); i++)
    {
        SubGraphModel* pSubModel = pModel->subGraph(i);
        Q_ASSERT(pSubModel);
        ZenoSubGraphView* pView = new ZenoSubGraphView;
        pView->setModel(pSubModel);
        addWidget(pView);
    }

    connect(m_model, SIGNAL(rowsRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsRemoved(const QModelIndex&, int, int)));
    connect(m_model, SIGNAL(rowsInserted(const QModelIndex&, int, int)),
        this, SLOT(onRowsInserted(const QModelIndex&, int, int)));
}

GraphsModel* ZenoGraphsWidget::model() const
{
    return m_model;
}

QList<QAction*> ZenoGraphsWidget::getCategoryActions(QPointF scenePos)
{
    NODE_CATES cates = m_model->getCates();
    QList<QAction *> acts;
    if (cates.isEmpty())
    {
        QAction *pAction = new QAction("ERROR: no descriptors loaded!");
        pAction->setEnabled(false);
        acts.push_back(pAction);
        return acts;
    }
    for (const NODE_CATE& cate : cates)
    {
        QAction *pAction = new QAction(cate.name);
        QMenu *pChildMenu = new QMenu;
        pChildMenu->setToolTipsVisible(true);
        for (const QString& name : cate.nodes)
        {
            QAction* pChildAction = pChildMenu->addAction(name);
            //todo: tooltip
            connect(pChildAction, &QAction::triggered, this, [=]() {
                onNewNodeCreated(name, scenePos);
            });
        }
        pAction->setMenu(pChildMenu);
        acts.push_back(pAction);
    }
    return acts;
}

void ZenoGraphsWidget::onNewNodeCreated(const QString& descName, const QPointF& pt)
{
    NODE_DESCS descs = m_model->descriptors();
    const NODE_DESC &desc = descs[descName];

    NODEITEM_PTR pItem(new PlainNodeItem);
    const QString &nodeid = UiHelper::generateUuid(descName);
    pItem->setData(nodeid, ROLE_OBJID);
    pItem->setData(descName, ROLE_OBJNAME);
    pItem->setData(NORMAL_NODE, ROLE_OBJTYPE);
    pItem->setData(QVariant::fromValue(desc.inputs), ROLE_INPUTS);
    pItem->setData(QVariant::fromValue(desc.outputs), ROLE_OUTPUTS);
    pItem->setData(QVariant::fromValue(desc.params), ROLE_PARAMETERS);
    pItem->setData(pt, ROLE_OBJPOS);

    SubGraphModel* pModel = m_model->currentGraph();
    int row = pModel->rowCount();
    pModel->appendItem(pItem);
}

void ZenoGraphsWidget::onRowsRemoved(const QModelIndex &parent, int first, int last)
{
    removeWidget(widget(first));
}

void ZenoGraphsWidget::onRowsInserted(const QModelIndex &parent, int first, int last)
{
    SubGraphModel *pSubModel = m_model->subGraph(first);
    Q_ASSERT(pSubModel);
    ZenoSubGraphView *pView = new ZenoSubGraphView;
    pView->setModel(pSubModel);
    addWidget(pView);
}