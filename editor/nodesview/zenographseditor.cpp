#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "zenosubnettreeview.h"
#include <model/graphsmodel.h>
#include <model/graphstreemodel.h>

#define USE_LISTVIEW_PANEL


ZenoGraphsEditor::ZenoGraphsEditor(QWidget* parent)
    : QWidget(parent)
    , m_pSubnetBtn(nullptr)
    , m_pSubnetList(nullptr)
    , m_pTabWidget(nullptr)
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
#else
    m_pSubnetTree = new ZenoSubnetTreeView;
    m_pSubnetTree->hide();
    pLayout->addWidget(m_pSubnetTree);
    //connect(m_pSubnetTree, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));
#endif

    m_pTabWidget = new ZenoGraphsTabWidget();
    pLayout->addWidget(m_pTabWidget);

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
    const QString& subgraphName = index.data().toString();
    m_pTabWidget->activate(subgraphName);
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
#else
    m_pSubnetTree->hide();
#endif
    m_pSideBar->hide();
    m_pTabWidget->clear();
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
