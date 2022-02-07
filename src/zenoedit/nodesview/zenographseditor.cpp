#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenographslayerwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "zenosubnettreeview.h"
#include <model/graphsmodel.h>
#include <model/graphstreemodel.h>
#include <model/modelrole.h>

#define USE_LISTVIEW_PANEL


ZenoGraphsEditor::ZenoGraphsEditor(QWidget* parent)
    : QWidget(parent)
    , m_pSubnetBtn(nullptr)
    , m_pSubnetList(nullptr)
    , m_pTabWidget(nullptr)
    , m_pLayerWidget(nullptr)
    , m_pSideBar(nullptr)
    , m_pSubnetTree(nullptr)
{
    QHBoxLayout* pLayout = new QHBoxLayout;

    m_pSideBar = new QWidget;

    QVBoxLayout* pVLayout = new QVBoxLayout;
    m_pSubnetBtn = new ZToolButton(
        ZToolButton::Opt_HasIcon | ZToolButton::Opt_HasText | ZToolButton::Opt_UpRight,
        QIcon(":/icons/subnetbtn.svg"),
        QSize(20, 20),
        "Subset",
        nullptr
    );
    m_pSubnetBtn->setBackgroundClr(QColor(36, 36, 36), QColor(36, 36, 36), QColor(36, 36, 36));
    m_pSubnetBtn->setCheckable(true);

    pVLayout->addWidget(m_pSubnetBtn);
    pVLayout->addStretch();
    pVLayout->setSpacing(0);
    pVLayout->setContentsMargins(0, 0, 0, 0);

    m_pSideBar->setLayout(pVLayout);
    QPalette pal = m_pSideBar->palette();
    pal.setColor(QPalette::Window, QColor(42, 42, 42));
    m_pSideBar->setAutoFillBackground(true);
    m_pSideBar->setPalette(pal);
    m_pSideBar->hide();

    pLayout->addWidget(m_pSideBar);


    QVBoxLayout* pLayout2 = new QVBoxLayout;

#ifdef USE_LISTVIEW_PANEL
    m_pSubnetList = new ZenoSubnetListPanel();
    m_pSubnetList->hide();
    pLayout->addWidget(m_pSubnetList);
    connect(m_pSubnetList, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));
    m_pTabWidget = new ZenoGraphsTabWidget();
    pLayout->addWidget(m_pTabWidget);
#else
    m_pSubnetTree = new ZenoSubnetTreeView;
    m_pSubnetTree->hide();
    pLayout->addWidget(m_pSubnetTree);

    m_pLayerWidget = new ZenoGraphsLayerWidget;
    connect(m_pSubnetTree, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));
    pLayout->addWidget(m_pLayerWidget);
#endif

    connect(m_pSubnetBtn, SIGNAL(clicked()), this, SLOT(onSubnetBtnClicked()));

    pLayout->setSpacing(1);
    pLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(pLayout);
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{
}

void ZenoGraphsEditor::onListItemActivated(const QModelIndex& index)
{
#ifdef USE_LISTVIEW_PANEL
    const QString& subgraphName = index.data().toString();
    m_pTabWidget->activate(subgraphName);
#else
    QSharedPointer<GraphsManagment> spGm = zenoApp->graphsManagment();
    GraphsModel* pModel = spGm->currentModel();
    QModelIndex idx = index;
    QString path;
    do
    {
        const QString& objName = idx.data().toString();
        if (pModel->subGraph(objName))
        {
            path = "/" + objName + path;
        }
        idx = idx.parent();
    } while (idx.isValid());
    m_pLayerWidget->resetPath(path);
#endif
}

void ZenoGraphsEditor::resetModel(GraphsModel* pModel)
{
#ifdef USE_LISTVIEW_PANEL
    m_pSubnetList->initModel(pModel);
    m_pTabWidget->resetModel(pModel);
    m_pSideBar->show();
    m_pSubnetBtn->setChecked(true);
    m_pSubnetList->show();
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onCurrentModelClear()));
#else
    GraphsTreeModel* pTreeModel = new GraphsTreeModel(pModel, this);
    pTreeModel->init(pModel);
    m_pSubnetTree->initModel(pTreeModel);
    m_pSideBar->show();
    m_pSubnetBtn->setChecked(true);
    m_pSubnetTree->show();
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onCurrentModelClear()));
#endif
}

void ZenoGraphsEditor::onCurrentModelClear()
{
#ifdef USE_LISTVIEW_PANEL
    m_pSubnetList->hide();
    m_pTabWidget->clear();
#else
    m_pSubnetTree->hide();
    m_pLayerWidget;
#endif
    m_pSideBar->hide();
    
}

void ZenoGraphsEditor::onSubnetBtnClicked()
{
    GraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (pModel == nullptr)
    {
        //open file dialog
    }
    else
    {
#ifdef USE_LISTVIEW_PANEL
        if (m_pSubnetBtn->isChecked())
        {
            m_pSubnetList->show();
        }
        else
        {
            m_pSubnetList->hide();
        }
#else

#endif
    }
}
