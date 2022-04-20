#include "zenographseditor.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "zenowelcomepage.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include <model/graphstreemodel.h>
#include <zenoui/model/modelrole.h>
#include <comctrl/zenocheckbutton.h>
#include <comctrl/ziconbutton.h>
#include <zenoui/style/zenostyle.h>
#include "zenomainwindow.h"
#include "nodesys/zenosubgraphview.h"
#include "ui_zenographseditor.h"
#include "nodesview/zsubnetlistitemdelegate.h"
#include "searchitemdelegate.h"
#include <zenoui/util/cihou.h>


ZenoGraphsEditor::ZenoGraphsEditor(ZenoMainWindow* pMainWin)
	: QWidget(nullptr)
	, m_mainWin(pMainWin)
    , m_model(nullptr)
    , m_searchOpts(SEARCHALL)
{
    initUI();
    initModel();
    initSignals();
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{
}

void ZenoGraphsEditor::initUI()
{
	m_ui = new Ui::GraphsEditor;
	m_ui->setupUi(this);

    m_ui->subnetBtn->setIcons(QIcon(":/icons/ic_sidemenu_subnet.svg"), QIcon(":/icons/ic_sidemenu_subnet_on.svg"));
    m_ui->treeviewBtn->setIcons(QIcon(":/icons/ic_sidemenu_tree.svg"), QIcon(":/icons/ic_sidemenu_tree_on.svg"));
    m_ui->searchBtn->setIcons(QIcon(":/icons/ic_sidemenu_search.svg"), QIcon(":/icons/ic_sidemenu_search_on.svg"));

    int _margin = ZenoStyle::dpiScaled(10);
    QMargins margins(_margin, _margin, _margin, _margin);
    QSize szIcons = ZenoStyle::dpiScaledSize(QSize(20, 20));

    m_ui->moreBtn->setIcons(szIcons, ":/icons/more.svg", ":/icons/more_on.svg");
    m_ui->btnSearchOpt->setIcons(szIcons, ":/icons/more.svg", ":/icons/more_on.svg");

    m_ui->subnetBtn->setSize(szIcons, margins);
    m_ui->treeviewBtn->setSize(szIcons, margins);
    m_ui->searchBtn->setSize(szIcons, margins);

    m_ui->stackedWidget->setCurrentWidget(m_ui->subnetPage);
    m_ui->splitter->setStretchFactor(1, 5);

    m_ui->mainStackedWidget->setCurrentWidget(m_ui->welcomePage);

    m_ui->graphsViewTab->setFont(QFont("HarmonyOS Sans", 12));  //bug in qss font setting.
    m_ui->searchEdit->setProperty("cssClass", "searchEdit");
}

void ZenoGraphsEditor::initModel()
{
    m_sideBarModel = new QStandardItemModel;

    QStandardItem* pItem = new QStandardItem;
	pItem->setData(Side_Subnet);
    m_sideBarModel->appendRow(pItem);

    pItem = new QStandardItem;
    pItem->setData(Side_Tree);
    m_sideBarModel->appendRow(pItem);

    pItem = new QStandardItem;
    pItem->setData(Side_Search);
    m_sideBarModel->appendRow(pItem);

    m_selection = new QItemSelectionModel(m_sideBarModel);
}

void ZenoGraphsEditor::initSignals()
{
	auto graphsMgr = zenoApp->graphsManagment();
	connect(graphsMgr.get(), SIGNAL(modelInited(IGraphsModel*)), this, SLOT(resetModel(IGraphsModel*)));

    connect(m_ui->subnetBtn, &ZenoCheckButton::toggled, this, &ZenoGraphsEditor::sideButtonToggled);
    connect(m_ui->treeviewBtn, &ZenoCheckButton::toggled, this, &ZenoGraphsEditor::sideButtonToggled);
    connect(m_ui->searchBtn, &ZenoCheckButton::toggled, this, &ZenoGraphsEditor::sideButtonToggled);

    connect(m_selection, &QItemSelectionModel::selectionChanged, this, &ZenoGraphsEditor::onSideBtnToggleChanged);
    connect(m_selection, &QItemSelectionModel::currentChanged, this, &ZenoGraphsEditor::onCurrentChanged);

    connect(m_ui->subnetList, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));
    connect(m_ui->subnetTree, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onTreeItemActivated(const QModelIndex&)));

	connect(m_ui->welcomePage, SIGNAL(newRequest()), m_mainWin, SLOT(onNewFile()));
	connect(m_ui->welcomePage, SIGNAL(openRequest()), m_mainWin, SLOT(openFileDialog()));

    connect(m_ui->moreBtn, SIGNAL(clicked()), this, SLOT(onSubnetOptionClicked()));
    connect(m_ui->btnSearchOpt, SIGNAL(clicked()), this, SLOT(onSearchOptionClicked()));
    connect(m_ui->graphsViewTab, &QTabWidget::tabCloseRequested, this, [=](int index) {
        m_ui->graphsViewTab->removeTab(index);
    });
    connect(m_ui->searchEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onSearchEdited(const QString&)));
    connect(m_ui->searchResView, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onSearchItemClicked(const QModelIndex&)));

    m_selection->setCurrentIndex(m_sideBarModel->index(0, 0), QItemSelectionModel::SelectCurrent);
}

void ZenoGraphsEditor::resetModel(IGraphsModel* pModel)
{
    if (!pModel)
    {
        onModelCleared();
        return;
    }

    auto mgr = zenoApp->graphsManagment();
    m_model = pModel;
    Q_ASSERT(m_model);

    GraphsTreeModel* pTreeModel = mgr->treeModel();
    m_ui->subnetTree->setModel(pTreeModel);
    m_ui->subnetList->setModel(pModel);

    m_ui->subnetList->setItemDelegate(new ZSubnetListItemDelegate(m_model, this));

    m_ui->mainStackedWidget->setCurrentWidget(m_ui->mainEditor);
    m_ui->graphsViewTab->clear();

    connect(pModel, &IGraphsModel::modelClear, this, &ZenoGraphsEditor::onModelCleared);
	connect(pModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)), this, SLOT(onSubGraphsToRemove(const QModelIndex&, int, int)));
	connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
	connect(pModel, SIGNAL(graphRenamed(const QString&, const QString&)), this, SLOT(onSubGraphRename(const QString&, const QString&)));

    activateTab("main");
}

void ZenoGraphsEditor::onModelCleared()
{
    m_ui->mainStackedWidget->setCurrentWidget(m_ui->welcomePage);
}

void ZenoGraphsEditor::onSubGraphsToRemove(const QModelIndex& parent, int first, int last)
{
	for (int r = first; r <= last; r++)
	{
		QModelIndex subgIdx = m_model->index(r, 0);
		const QString& name = m_model->name(subgIdx);
		int idx = tabIndexOfName(name);
		m_ui->graphsViewTab->removeTab(idx);
	}
}

void ZenoGraphsEditor::onModelReset()
{
	m_ui->graphsViewTab->clear();
    m_model = nullptr;
}

void ZenoGraphsEditor::onSubGraphRename(const QString& oldName, const QString& newName)
{
	int idx = tabIndexOfName(oldName);
	if (idx != -1)
	{
		QTabBar* pTabBar = m_ui->graphsViewTab->tabBar();
		pTabBar->setTabText(idx, newName);
	}
}

void ZenoGraphsEditor::onSearchOptionClicked()
{
	QMenu* pOptionsMenu = new QMenu;

	QAction* pNode = new QAction(tr("Node"));
    pNode->setCheckable(true);
    pNode->setChecked(m_searchOpts & SEARCH_NODE);

	QAction* pSubnet = new QAction(tr("Subnet"));
    pSubnet->setCheckable(true);
    pSubnet->setChecked(m_searchOpts & SEARCH_SUBNET);

	QAction* pAnnotation = new QAction(tr("Annotation"));
    pAnnotation->setCheckable(true);
    pAnnotation->setEnabled(false);

	QAction* pWrangle = new QAction(tr("wrangle snippet"));
    pWrangle->setCheckable(true);
    pWrangle->setChecked(m_searchOpts & SEARCH_WRANGLE);

	pOptionsMenu->addAction(pNode);
	pOptionsMenu->addAction(pSubnet);
	pOptionsMenu->addAction(pAnnotation);
	pOptionsMenu->addAction(pWrangle);

	connect(pNode, &QAction::triggered, this, [=](bool bChecked) {
        if (bChecked)
            m_searchOpts |= SEARCH_NODE;
        else
            m_searchOpts &= (~(int)SEARCH_NODE);
	});

	connect(pSubnet, &QAction::triggered, this, [=](bool bChecked) {
		if (bChecked)
			m_searchOpts |= SEARCH_SUBNET;
		else
			m_searchOpts &= (~(int)SEARCH_SUBNET);
		});

	connect(pAnnotation, &QAction::triggered, this, [=](bool bChecked) {
		if (bChecked)
			m_searchOpts |= SEARCH_ANNO;
		else
			m_searchOpts &= (~(int)SEARCH_ANNO);
		});

	connect(pWrangle, &QAction::triggered, this, [=](bool bChecked) {
		if (bChecked)
			m_searchOpts |= SEARCH_WRANGLE;
		else
			m_searchOpts &= (~(int)SEARCH_WRANGLE);
		});

	pOptionsMenu->exec(QCursor::pos());
	pOptionsMenu->deleteLater();
}

void ZenoGraphsEditor::onSubnetOptionClicked()
{
    QMenu* pOptionsMenu = new QMenu;

	QAction* pCreate = new QAction(tr("create subnet"));
	QAction* pSubnetMap = new QAction("subnet map");
	QAction* pImpFromFile = new QAction("import from local file");
	QAction* pImpFromSys = new QAction("import system subnet");

    pOptionsMenu->addAction(pCreate);
    pOptionsMenu->addAction(pSubnetMap);
    pOptionsMenu->addSeparator();
    pOptionsMenu->addAction(pImpFromFile);
    pOptionsMenu->addAction(pImpFromSys);

    connect(pCreate, &QAction::triggered, this, [=]() {
        bool bOk = false;
        QString newSubgName = QInputDialog::getText(this, tr("create subnet"), tr("new subgraph name:")
            , QLineEdit::Normal, "subgraph name", &bOk);
        if (bOk) {
            m_model->newSubgraph(newSubgName);
        }
	});
	connect(pSubnetMap, &QAction::triggered, this, [=]() {

		});
	connect(pImpFromFile, &QAction::triggered, this, [=]() {

		});
	connect(pImpFromSys, &QAction::triggered, this, [=]() {

		});

    pOptionsMenu->exec(QCursor::pos());
    pOptionsMenu->deleteLater();
}

void ZenoGraphsEditor::sideButtonToggled(bool bToggled)
{
    QObject* pBtn = sender();

    QModelIndex idx;
    if (pBtn == m_ui->subnetBtn)
    {
        idx = m_sideBarModel->match(m_sideBarModel->index(0, 0), Qt::UserRole + 1, Side_Subnet)[0];
    }
    else if (pBtn == m_ui->treeviewBtn)
    {
        idx = m_sideBarModel->match(m_sideBarModel->index(0, 0), Qt::UserRole + 1, Side_Tree)[0];
    }
    else if (pBtn == m_ui->searchBtn)
    {
        idx = m_sideBarModel->match(m_sideBarModel->index(0, 0), Qt::UserRole + 1, Side_Search)[0];
    }
    else
    {
        Q_ASSERT(false);
    }

    if (bToggled)
        m_selection->setCurrentIndex(idx, QItemSelectionModel::SelectCurrent);
    else
        m_selection->clearCurrentIndex();
}

void ZenoGraphsEditor::onSideBtnToggleChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
}

void ZenoGraphsEditor::onCurrentChanged(const QModelIndex& current, const QModelIndex& previous)
{
	if (previous.isValid())
	{
        int sideBar = current.data(Qt::UserRole + 1).toInt();
		switch (previous.data(Qt::UserRole + 1).toInt())
		{
		case Side_Subnet: m_ui->subnetBtn->setChecked(false); break;
		case Side_Tree: m_ui->treeviewBtn->setChecked(false); break;
		case Side_Search: m_ui->searchBtn->setChecked(false); break;
		}
	}

	if (current.isValid())
	{
        m_ui->stackedWidget->show();
		int sideBar = current.data(Qt::UserRole + 1).toInt();
        switch (sideBar)
        {
            case Side_Subnet:
            {
                m_ui->subnetBtn->setChecked(true);
                m_ui->stackedWidget->setCurrentWidget(m_ui->subnetPage);
                break;
            }
            case Side_Tree:
            {
                m_ui->treeviewBtn->setChecked(true);
                m_ui->stackedWidget->setCurrentWidget(m_ui->treePage);
                break;
            }
            case Side_Search:
            {
                m_ui->searchBtn->setChecked(true);
                m_ui->stackedWidget->setCurrentWidget(m_ui->searchPage);
                break;
            }
        }
	}
    else
    {
        m_ui->stackedWidget->hide();
    }
}

int ZenoGraphsEditor::tabIndexOfName(const QString& subGraphName)
{
	for (int i = 0; i < m_ui->graphsViewTab->count(); i++)
	{
		if (m_ui->graphsViewTab->tabText(i) == subGraphName)
		{
			return i;
		}
	}
	return -1;
}

void ZenoGraphsEditor::onListItemActivated(const QModelIndex& index)
{
	const QString& subgraphName = index.data().toString();
    activateTab(subgraphName);
}

void ZenoGraphsEditor::activateTab(const QString& subGraphName, const QString& path, const QString& objId)
{
	auto graphsMgm = zenoApp->graphsManagment();
	IGraphsModel* pModel = graphsMgm->currentModel();

    if (!pModel->index(subGraphName).isValid())
        return;

	int idx = tabIndexOfName(subGraphName);
	if (idx == -1)
	{
		const QModelIndex& subgIdx = pModel->index(subGraphName);
		ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(pModel->scene(subgIdx));
		Q_ASSERT(pScene);

        ZenoSubGraphView* pView = new ZenoSubGraphView;
		pView->initScene(pScene);

		idx = m_ui->graphsViewTab->addTab(pView, subGraphName);

        connect(pView, &ZenoSubGraphView::pathUpdated, this, [=](QString newPath) {
            QStringList L = newPath.split("/", QtSkipEmptyParts);
            QString subgName = L.last();
            activateTab(subgName, newPath);
        });
	}
	m_ui->graphsViewTab->setCurrentIndex(idx);

    ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
    Q_ASSERT(pView);
    pView->resetPath(path, subGraphName, objId);
}

void ZenoGraphsEditor::onTreeItemActivated(const QModelIndex& index)
{
	QModelIndex idx = index;

	const QString& objId = idx.data(ROLE_OBJID).toString();
	QString path, subgName;
	if (!idx.parent().isValid())
	{
        subgName = idx.data(ROLE_OBJNAME).toString();
		path = "/" + subgName;
	}
	else
	{
		idx = idx.parent();
        subgName = idx.data(ROLE_OBJNAME).toString();

		while (idx.isValid())
		{
			QString objName = idx.data(ROLE_OBJNAME).toString();
			path = "/" + objName + path;
			idx = idx.parent();
		}
	}

    activateTab(subgName, path, objId);
}

void ZenoGraphsEditor::onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx)
{
    const QString& subgName = nodeIdx.data(ROLE_OBJNAME).toString();
    activateTab(subgName);
}

void ZenoGraphsEditor::onSearchEdited(const QString& content)
{
    QList<SEARCH_RESULT> results = m_model->search(content, m_searchOpts);

    QStandardItemModel* pModel = new QStandardItemModel(this);

    for (SEARCH_RESULT res : results)
    {
        if (res.type == SEARCH_SUBNET)
		{
            QString subgName = res.targetIdx.data(ROLE_OBJNAME).toString();
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, subgName, 1, Qt::MatchExactly);
            if (lst.size() == 0)
            {
				//add subnet
				QStandardItem* pItem = new QStandardItem(subgName + " (Subnet)");
				pItem->setData(subgName, ROLE_OBJNAME);
                pItem->setData(res.targetIdx.data(ROLE_OBJID).toString(), ROLE_OBJID);
				pModel->appendRow(pItem);
            }
        }
        else if (res.type == SEARCH_NODE)
        {
            QString subgName = res.subgIdx.data(ROLE_OBJNAME).toString();
			QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, subgName, 1, Qt::MatchExactly);

            QStandardItem* parentItem = nullptr;
            if (lst.size() == 0)
            {
				//add subnet
                parentItem = new QStandardItem(subgName + " (Subnet)");
                parentItem->setData(subgName, ROLE_OBJNAME);
				pModel->appendRow(parentItem);
            }
            else
            {
                Q_ASSERT(lst.size() == 1);
                parentItem = pModel->itemFromIndex(lst[0]);
            }

            QString nodeName = res.targetIdx.data(ROLE_OBJNAME).toString();
            QStandardItem* pItem = new QStandardItem(nodeName);
            pItem->setData(nodeName, ROLE_OBJNAME);
            pItem->setData(res.targetIdx.data(ROLE_OBJID).toString(), ROLE_OBJID);
		    parentItem->appendRow(pItem);
        }
    }

    m_ui->searchResView->setModel(pModel);
    m_ui->searchResView->setItemDelegate(new SearchItemDelegate(content));
    m_ui->searchResView->expandAll();
}

void ZenoGraphsEditor::onSearchItemClicked(const QModelIndex& index)
{
    QString objId = index.data(ROLE_OBJID).toString();
    if (index.parent().isValid())
    {
        QString parentId = index.parent().data(ROLE_OBJID).toString();
        QString subgName = index.parent().data(ROLE_OBJNAME).toString();
        activateTab(subgName, "", objId);
    }
    else
    {
        QString subgName = index.data(ROLE_OBJNAME).toString();
        activateTab(subgName);
    }
}

void ZenoGraphsEditor::onMenuActionTriggered(QAction* pAction)
{
    const QString& text = pAction->text();
    if (text == "Collaspe")
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        Q_ASSERT(pView);
        QModelIndex subgIdx = pView->scene()->subGraphIndex();
        m_model->collaspe(subgIdx);
    }
    else if (text == "Expand")
	{
		ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
		Q_ASSERT(pView);
		QModelIndex subgIdx = pView->scene()->subGraphIndex();
		m_model->expand(subgIdx);
    }
}
