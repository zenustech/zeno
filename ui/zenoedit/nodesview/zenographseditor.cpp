#include "zenographseditor.h"
#include "zenographstabwidget.h"
#include "zenographslayerwidget.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "zenowelcomepage.h"
#include "graphsmanagment.h"
#include "zenosubnettreeview.h"
#include "model/graphsmodel.h"
#include <model/graphstreemodel.h>
#include <zenoui/model/modelrole.h>
#include "zenomainwindow.h"
#include "zenosubnetpanel.h"


ZenoGraphsEditor::ZenoGraphsEditor(ZenoMainWindow* pMainWin)
    : QWidget(nullptr)
    , m_pSubnetBtn(nullptr)
    , m_pSubnetPanel(nullptr)
    , m_pTabWidget(nullptr)
    , m_pLayerWidget(nullptr)
    , m_pSideBar(nullptr)
    , m_bListView(true)
    , m_pViewBtn(nullptr)
    , m_welcomePage(nullptr)
    , m_mainWin(pMainWin)
{
    QHBoxLayout* pLayout = new QHBoxLayout;

    m_pSideBar = new QWidget;

    QVBoxLayout* pVLayout = new QVBoxLayout;
    m_pSubnetBtn = new ZToolButton(
        ZToolButton::Opt_HasIcon | /*ZToolButton::Opt_HasText | */ZToolButton::Opt_UpRight,
        QIcon(":/icons/subnetbtn.svg"),
		QSize(20, 20)/*,
		"Subset",
		nullptr*/
    );
    m_pSubnetBtn->setBackgroundClr(QColor(36, 36, 36), QColor(36, 36, 36), QColor(36, 36, 36));
    m_pSubnetBtn->setCheckable(true);

    m_pViewBtn = new ZToolButton(
        ZToolButton::Opt_HasIcon,
        QIcon(":/icons/treeview.svg"),
        QSize(20, 20)
    );
    m_pViewBtn->setBackgroundClr(QColor(36, 36, 36), QColor(36, 36, 36), QColor(36, 36, 36));
    m_pViewBtn->setCheckable(true);

    pVLayout->addWidget(m_pSubnetBtn);
    pVLayout->addWidget(m_pViewBtn);
    pVLayout->addStretch();
    pVLayout->setSpacing(1);
    pVLayout->setContentsMargins(0, 0, 0, 0);

    m_pSideBar->setLayout(pVLayout);
    QPalette pal = m_pSideBar->palette();
    pal.setColor(QPalette::Window, QColor(42, 42, 42));
    m_pSideBar->setAutoFillBackground(true);
    m_pSideBar->setPalette(pal);
    m_pSideBar->hide();

    pLayout->addWidget(m_pSideBar);

    QVBoxLayout* pLayout2 = new QVBoxLayout;

    m_pSubnetPanel = new ZenoSubnetPanel();
    m_pSubnetPanel->hide();
    pLayout->addWidget(m_pSubnetPanel);

    m_pTabWidget = new ZenoGraphsTabWidget;
    m_pLayerWidget = new ZenoGraphsLayerWidget;

    pLayout->addWidget(m_pTabWidget);
    pLayout->addWidget(m_pLayerWidget);

    connect(m_pSubnetPanel, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onItemActivated(const QModelIndex&)));
    connect(m_pSubnetPanel, SIGNAL(graphToBeActivated(const QString&)), m_pTabWidget, SLOT(activate(const QString&)));
    connect(m_pSubnetBtn, SIGNAL(clicked()), this, SLOT(onSubnetBtnClicked()));
    connect(m_pViewBtn, SIGNAL(clicked()), this, SLOT(onViewBtnClicked()));

    pLayout->setSpacing(1);
    pLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(pLayout);

    m_pViewBtn->setChecked(!m_bListView);
    m_pTabWidget->setVisible(m_bListView);
    m_pLayerWidget->setVisible(!m_bListView);

    m_welcomePage = new ZenoWelcomePage;
    pLayout->addWidget(m_welcomePage);
    m_welcomePage->setVisible(true);
    m_pSubnetPanel->setVisible(false);
    m_pTabWidget->setVisible(false);
    m_pLayerWidget->setVisible(false);
    m_pSubnetBtn->setVisible(false);
    m_pViewBtn->setVisible(false);

    connect(m_welcomePage, SIGNAL(newRequest()), this, SLOT(onNewFile()));
    connect(m_welcomePage, SIGNAL(openRequest()), m_mainWin, SLOT(openFileDialog()));
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{
}

void ZenoGraphsEditor::onNewFile()
{
    
}

void ZenoGraphsEditor::onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx)
{
    if (m_bListView)
    {
        // subgraph node.
        const QString& subgName = nodeIdx.data(ROLE_OBJNAME).toString();
        m_pTabWidget->activate(subgName);
    }
    else
    {
		// subgraph node.
		const QString& subgName = nodeIdx.data(ROLE_OBJNAME).toString();
        QString path = m_pLayerWidget->path();
        path += "/" + subgName;
        m_pLayerWidget->resetPath(path, "");
    }
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
        IGraphsModel* pModel = spGm->currentModel();
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

void ZenoGraphsEditor::resetModel(IGraphsModel* pModel)
{
    m_pSideBar->show();
    m_pSubnetBtn->show();
    m_pViewBtn->show();
    if (pModel)
    {
		m_pSubnetPanel->initModel(pModel);
		if (m_bListView)
		{
			m_pTabWidget->resetModel(pModel);
            m_pTabWidget->show();
		}
		else
		{
			m_pLayerWidget->show();
		}

		m_pSubnetBtn->setChecked(true);
		m_pSubnetPanel->show();
		connect(pModel, SIGNAL(modelClear()), this, SLOT(onCurrentModelClear()));
    }
	m_welcomePage->setVisible(false);
    connect(pModel, &QAbstractItemModel::rowsInserted, this, &ZenoGraphsEditor::onGraphsItemInserted);
    connect(pModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &ZenoGraphsEditor::onGraphsItemAboutToBeRemoved);
}

void ZenoGraphsEditor::onGraphsItemInserted(const QModelIndex& parent, int first, int last)
{
    /*
	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (m_bListView)
    {
		const QString& subgraphName = idx.data(ROLE_OBJNAME).toString();
		m_pTabWidget->activate(subgraphName);
    }
    */
}

void ZenoGraphsEditor::onGraphsItemAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
}

void ZenoGraphsEditor::onCurrentModelClear()
{
    m_pSubnetPanel->hide();
    m_pTabWidget->clear();
    m_pLayerWidget->clear();
    m_pSideBar->hide();

	m_welcomePage->setVisible(true);
	m_pSubnetPanel->setVisible(false);
	m_pTabWidget->setVisible(false);
	m_pLayerWidget->setVisible(false);
	m_pSubnetBtn->setVisible(false);
	m_pViewBtn->setVisible(false);
}

void ZenoGraphsEditor::onViewBtnClicked()
{
	if (m_pViewBtn->isChecked())
	{
        m_bListView = false;
	}
	else
	{
        m_bListView = true;
	}
    m_pSubnetPanel->setViewWay(m_bListView);
	m_pTabWidget->setVisible(m_bListView);
	m_pLayerWidget->setVisible(!m_bListView);
}

void ZenoGraphsEditor::onSubnetBtnClicked()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (pModel == nullptr)
    {
        //open file dialog
    }
    else
    {
        if (m_pSubnetBtn->isChecked())
        {
            m_pSubnetPanel->show();
        }
        else
        {
            m_pSubnetPanel->hide();
        }
    }
}
