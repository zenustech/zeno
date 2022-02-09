#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenographslayerwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "zenosubnettreeview.h"
#include <zenoui/model/graphsmodel.h>
#include <model/graphstreemodel.h>
#include <zenoui/model/modelrole.h>


ZenoGraphsEditor::ZenoGraphsEditor(QWidget* parent)
    : QWidget(parent)
    , m_pSubnetBtn(nullptr)
    , m_pSubnetList(nullptr)
    , m_pTabWidget(nullptr)
    , m_pLayerWidget(nullptr)
    , m_pSideBar(nullptr)
    , m_bListView(false)
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

    m_pSubnetList = new ZenoSubnetListPanel();
    m_pSubnetList->hide();
    pLayout->addWidget(m_pSubnetList);

    m_pTabWidget = new ZenoGraphsTabWidget;
    m_pLayerWidget = new ZenoGraphsLayerWidget;

    pLayout->addWidget(m_pTabWidget);
    pLayout->addWidget(m_pLayerWidget);

    connect(m_pSubnetList, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onItemActivated(const QModelIndex&)));
    connect(m_pSubnetBtn, SIGNAL(clicked()), this, SLOT(onSubnetBtnClicked()));

    pLayout->setSpacing(1);
    pLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(pLayout);

    m_pTabWidget->setVisible(m_bListView);
    m_pLayerWidget->setVisible(!m_bListView);
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{
}

void ZenoGraphsEditor::onItemActivated(const QModelIndex& index)
{
    if (m_bListView)
	{
		const QString& subgraphName = index.data().toString();
		m_pTabWidget->activate(subgraphName);
    }
    else
    {
        QSharedPointer<GraphsManagment> spGm = zenoApp->graphsManagment();
        GraphsModel* pModel = spGm->currentModel();
        QModelIndex idx = index;

        const QString& objId = idx.data(ROLE_OBJID).toString();
        QString path;
        if (!idx.parent().isValid())
        {
            path = "/" + idx.data(ROLE_OBJNAME).toString();
        }
        else
        {
            idx = idx.parent();
            while (idx.isValid())
            {
                QString objName = idx.data(ROLE_OBJNAME).toString();
                path = "/" + objName + path;
                idx = idx.parent();
            }
        }
        m_pLayerWidget->resetPath(path, objId);
    }
}

void ZenoGraphsEditor::resetModel(GraphsModel* pModel)
{
    m_pSubnetList->initModel(pModel);
    if (m_bListView)
    {
		m_pTabWidget->resetModel(pModel);
    }
    else
    {
        m_pLayerWidget;
    }
	m_pSideBar->show();
	m_pSubnetBtn->setChecked(true);
	m_pSubnetList->show();
    connect(pModel, SIGNAL(modelReset()), this, SLOT(onCurrentModelClear()));
}

void ZenoGraphsEditor::onCurrentModelClear()
{
    m_pSubnetList->hide();
    m_pTabWidget->clear();
    m_pLayerWidget;
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
        if (m_pSubnetBtn->isChecked())
        {
            m_pSubnetList->show();
        }
        else
        {
            m_pSubnetList->hide();
        }
    }
}
