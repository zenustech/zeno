#include "zenosubgraphscene.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/subgraphmodel.h>
#include <zenoui/model/modelrole.h>
#include "zenosubgraphview.h"
#include "zenographswidget.h"
#include <zenoui/util/uihelper.h>
#include <zeno/utils/log.h>


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

    for (int i = 0; i < pModel->rowCount(); i++)
    {
        ZenoSubGraphView* pView = new ZenoSubGraphView;
        //pView->initScene(pSubModel->scene());
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
{//completely not called
    NODE_DESCS descs = m_model->descriptors();
    const NODE_DESC &desc = descs[descName];

    const QString &nodeid = UiHelper::generateUuid(descName);
    NODE_DATA node;
    node[ROLE_OBJID] = nodeid;
    node[ROLE_OBJNAME] = descName;
    node[ROLE_NODETYPE] = NORMAL_NODE;
    node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
    node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
    node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
    node[ROLE_OBJPOS] = pt;

    SubGraphModel* pModel = m_model->currentGraph();
    int row = pModel->rowCount();
    pModel->appendItem(node, true);
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
    //pView->initScene(pSubModel->scene());
    addWidget(pView);
}

void ZenoGraphsWidget::clear()
{

}
