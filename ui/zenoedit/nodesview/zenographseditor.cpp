#include "zenographseditor.h"
#include "zenosubnetlistview.h"
#include <comctrl/ztoolbutton.h>
#include "zenoapplication.h"
#include "../nodesys/zenosubgraphscene.h"
#include "zenowelcomepage.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/viewparammodel.h>
#include "variantptr.h"
#include <comctrl/zenocheckbutton.h>
#include <comctrl/ziconbutton.h>
#include <zenoui/style/zenostyle.h>
#include "zenomainwindow.h"
#include "nodesys/zenosubgraphview.h"
#include "ui_zenographseditor.h"
#include "nodesview/zsubnetlistitemdelegate.h"
#include "searchitemdelegate.h"
#include "common_def.h"
#include "startup/zstartup.h"
#include "util/log.h"
#include "settings/zsettings.h"
#include "dialog/zeditparamlayoutdlg.h"
#include <zenomodel/include/nodesmgr.h>
#include "settings/zenosettingsmanager.h"
#include <zenoui/comctrl/zpathedit.h>
#include <zenomodel/include/uihelper.h>


ZenoGraphsEditor::ZenoGraphsEditor(ZenoMainWindow* pMainWin)
    : QWidget(nullptr)
    , m_mainWin(pMainWin)
    , m_pNodeModel(nullptr)
    , m_searchOpts(SEARCHALL)
{
    initUI();
    initModel();
    initSignals();

    auto graphsMgm = zenoApp->graphsManagment();
    if (graphsMgm) {
        IGraphsModel* pModel = graphsMgm->currentModel();
        if (pModel) {
            resetModel(pModel, graphsMgm->sharedSubgraphs());
        }
    }
}

ZenoGraphsEditor::~ZenoGraphsEditor()
{
}

void ZenoGraphsEditor::initUI()
{
    m_ui = new Ui::GraphsEditor;
    m_ui->setupUi(this);

    int _margin = ZenoStyle::dpiScaled(10);
    QMargins margins(_margin, _margin, _margin, _margin);
    QSize szIcons = ZenoStyle::dpiScaledSize(QSize(20, 20));
    m_ui->moreBtn->setIcons(szIcons, ":/icons/more.svg", ":/icons/more_on.svg");

    m_ui->splitter->setStretchFactor(1, 5);

    m_ui->mainStackedWidget->setCurrentWidget(m_ui->welcomeScrollPage);
    m_ui->stackedWidget->setCurrentIndex(0);

    QFont font = zenoApp->font();
    font.setPointSize(10);
    m_ui->graphsViewTab->setFont(font);  //bug in qss font setting.
    m_ui->graphsViewTab->tabBar()->setDrawBase(false);
    m_ui->graphsViewTab->setIconSize(ZenoStyle::dpiScaledSize(QSize(20,20)));
    m_ui->searchEdit->setProperty("cssClass", "searchEditor");
    m_ui->graphsViewTab->setProperty("cssClass", "graphicsediter");
    m_ui->graphsViewTab->tabBar()->setProperty("cssClass", "graphicsediter");
    m_ui->graphsViewTab->tabBar()->installEventFilter(this);
    m_ui->graphsViewTab->setUsesScrollButtons(true);
    m_ui->btnSearchOpt->setIcons(":/icons/collaspe.svg", ":/icons/collaspe.svg");
    initRecentFiles();
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
    connect(graphsMgr, SIGNAL(modelInited(IGraphsModel*, IGraphsModel*)),
            this, SLOT(resetModel(IGraphsModel*, IGraphsModel*)));
    connect(graphsMgr->logModel(), &QStandardItemModel::rowsInserted, this, &ZenoGraphsEditor::onLogInserted);

    connect(m_selection, &QItemSelectionModel::selectionChanged, this, &ZenoGraphsEditor::onSideBtnToggleChanged);
    connect(m_selection, &QItemSelectionModel::currentChanged, this, &ZenoGraphsEditor::onCurrentChanged);

    //connect(m_ui->subnetList, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onListItemActivated(const QModelIndex&)));
    //connect(m_ui->subnetTree, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onTreeItemActivated(const QModelIndex&)));

    connect(m_ui->welcomePage, SIGNAL(newRequest()), m_mainWin, SLOT(onNewFile()));
    connect(m_ui->welcomePage, SIGNAL(openRequest()), m_mainWin, SLOT(openFileDialog()));

    connect(m_ui->moreBtn, SIGNAL(clicked()), this, SLOT(onSubnetOptionClicked()));
    connect(m_ui->btnSearchOpt, SIGNAL(clicked()), this, SLOT(onSearchOptionClicked()));
    connect(m_ui->graphsViewTab, &QTabWidget::tabCloseRequested, this, [=](int index) {
        zenoApp->graphsManagment()->removeScene(m_ui->graphsViewTab->tabText(index));
        m_ui->graphsViewTab->removeTab(index);
    });
    connect(m_ui->searchEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onSearchEdited(const QString&)));
    connect(m_ui->searchResView, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onSearchItemClicked(const QModelIndex&)));

    auto& inst = ZenoSettingsManager::GetInstance();
    if (!inst.getValue("zencache-enable").isValid())
    {
        ZenoSettingsManager::GetInstance().setValue("zencache-enable", true);
    }

    //m_selection->setCurrentIndex(m_sideBarModel->index(0, 0), QItemSelectionModel::SelectCurrent);
}

void ZenoGraphsEditor::initRecentFiles()
{
    m_ui->welcomePage->initRecentFiles();
}

void ZenoGraphsEditor::resetModel(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphs)
{
    if (!pNodeModel)
    {
        onModelCleared();
        return;
    }

    m_pNodeModel = pNodeModel;
    m_pSubgraphs = pSubgraphs;
    ZASSERT_EXIT(m_pNodeModel);

    SubListSortProxyModel* treeProxyModel = new SubListSortProxyModel(this);
    treeProxyModel->setSourceModel(pNodeModel->implModel());
    treeProxyModel->setDynamicSortFilter(true);
    m_ui->subnetTree->setModel(treeProxyModel);
    m_ui->subnetTree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeProxyModel->sort(0, Qt::AscendingOrder);

    SubListSortProxyModel*proxyModel = new SubListSortProxyModel(this);
    proxyModel->setSourceModel(pSubgraphs);
    proxyModel->setDynamicSortFilter(true);
    m_ui->subnetList->setModel(proxyModel);
    proxyModel->sort(0, Qt::AscendingOrder);
    m_ui->subnetList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(m_ui->subnetTree->selectionModel(), &QItemSelectionModel::selectionChanged, this, &ZenoGraphsEditor::onTreeItemSelectionChanged);

    ZSubnetListItemDelegate *delegate = new ZSubnetListItemDelegate(m_pSubgraphs, this);
    connect(delegate, &ZSubnetListItemDelegate::subgrahSyncSignal, m_pNodeModel, &IGraphsModel::onSubgrahSync);
    m_ui->subnetList->setItemDelegate(delegate);
    connect(m_ui->subnetList->selectionModel(), &QItemSelectionModel::selectionChanged, this, [=]() {
        QModelIndexList lst = m_ui->subnetList->selectionModel()->selectedIndexes();
        delegate->setSelectedIndexs(lst);
        if (lst.size() == 1) 
        {
            onListItemActivated(lst.first());
        }
    });

    m_ui->mainStackedWidget->setCurrentWidget(m_ui->mainEditor);
    m_ui->graphsViewTab->clear();

    connect(m_pNodeModel, &IGraphsModel::modelClear, this, &ZenoGraphsEditor::onModelCleared);
    connect(m_pNodeModel, &IGraphsModel::_rowsAboutToBeRemoved, this, [=](const QModelIndex& subgIdex, const QModelIndex& parent, int first, int end) {
        onSubGraphsToRemove(m_pNodeModel, parent, first, end);
    });
    connect(m_pSubgraphs, &QAbstractItemModel::rowsAboutToBeRemoved, this, [=](const QModelIndex& parent, int first, int end)
    {
        onSubGraphsToRemove(m_pSubgraphs, parent, first, end);
    });

    connect(m_pNodeModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
    connect(m_pSubgraphs, SIGNAL(graphRenamed(const QString&, const QString &)), this,
            SLOT(onSubGraphRename(const QString &, const QString &)));

    activateTabOfTree("/main");
}

void ZenoGraphsEditor::onModelCleared()
{
    m_ui->mainStackedWidget->setCurrentWidget(m_ui->welcomeScrollPage);
    m_ui->searchEdit->clear();
}

void ZenoGraphsEditor::onSubGraphsToRemove(const IGraphsModel* pModel, const QModelIndex& parent, int first, int last)
{
    for (int r = first; r <= last; r++)
    {
        if (pModel == m_pSubgraphs)
        {
            const QModelIndex& subgIdx = m_pSubgraphs->index(r, 0);
            const QString& name = subgIdx.data(ROLE_OBJNAME).toString();
            int idx = tabIndexOfName(name);
            if (idx >= 0)
            {
                zenoApp->graphsManagment()->removeScene(name);
                m_ui->graphsViewTab->removeTab(idx);
            }
        }
        else
        {
            const QModelIndex& subgIdx = m_pNodeModel->index(r, 0, parent);
            if (!m_pNodeModel->IsSubGraphNode(subgIdx))
                continue;
            const QString& path = subgIdx.data(ROLE_OBJPATH).toString();
            int count = m_ui->graphsViewTab->count();
            for (int i = count - 1; i >= 0; i--)
            {
                const QString tabText = m_ui->graphsViewTab->tabText(i);
                if (tabText.contains(path, Qt::CaseInsensitive))
                {
                    zenoApp->graphsManagment()->removeScene(tabText);
                    m_ui->graphsViewTab->removeTab(i);
                }
            }
        }
    }
}

bool ZenoGraphsEditor::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == m_ui->graphsViewTab->tabBar() && event->type() == QEvent::Paint)
    {
        for (int idx = 0; idx < m_ui->graphsViewTab->count(); idx++)
        {
            QRectF rec = m_ui->graphsViewTab->tabBar()->tabRect(idx);
            QPainter painter(m_ui->graphsViewTab->tabBar());
            QColor color;
            QString text = m_ui->graphsViewTab->tabBar()->tabText(idx);
            if (m_pSubgraphs->index(text).isValid())
            {
                if (m_ui->graphsViewTab->currentIndex() == idx)
                    color = QColor("#0064A8");
                else
                    color = QColor("#4578AC");
            }
            else
            {
                if (m_ui->graphsViewTab->currentIndex() == idx)
                    color = QColor("#22252C");
                else
                    color = QColor("#2D3239");
            }
            painter.setBrush(color);
            painter.drawRect(rec);
            qreal y = rec.y() + (rec.height() - ZenoStyle::dpiScaled(20)) / 2;
            QIcon icon = m_ui->graphsViewTab->tabBar()->tabIcon(idx);
            painter.drawPixmap(QPointF(rec.x() + ZenoStyle::dpiScaled(4), y), icon.pixmap(ZenoStyle::dpiScaledSize(QSize(20, 20))));
            QPen pen(Qt::white);
            pen.setWidth(1);
            painter.setPen(pen);
            rec.adjust(ZenoStyle::dpiScaled(28), y, 0, 0);
            painter.drawText(rec, text);
        }
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void ZenoGraphsEditor::onModelReset()
{
    m_ui->graphsViewTab->clear();
    m_pNodeModel = nullptr;
}

void ZenoGraphsEditor::onSubGraphRename(const QString& oldName, const QString& newName)
{
    int idx = tabIndexOfName(oldName);
    if (idx != -1)
    {
        QTabBar* pTabBar = m_ui->graphsViewTab->tabBar();
        pTabBar->setTabText(idx, newName);
    }
    m_pNodeModel->renameSubGraph(oldName, newName);
}

void ZenoGraphsEditor::onSearchOptionClicked()
{
    QMenu* pOptionsMenu = new QMenu;

    QAction* pNode = new QAction(tr("Node Name"));
    pNode->setCheckable(true);
    pNode->setChecked(m_searchOpts & SEARCH_NODECLS);

    QAction* pNodeID = new QAction(tr("Node ID"));
    pNodeID->setCheckable(true);
    pNodeID->setChecked(m_searchOpts & SEARCH_NODEID);

    QAction* pSubnet = new QAction(tr("Subnet"));
    pSubnet->setCheckable(true);
    pSubnet->setChecked(m_searchOpts & SEARCH_SUBNET);

    QAction* pAnnotation = new QAction(tr("Annotation"));
    pAnnotation->setCheckable(true);
    pAnnotation->setEnabled(false);

    QAction* pWrangle = new QAction(tr("Parameter"));
    pWrangle->setCheckable(true);
    pWrangle->setChecked(m_searchOpts & SEARCH_ARGS);

    pOptionsMenu->addAction(pNode);
    pOptionsMenu->addAction(pNodeID);
    pOptionsMenu->addAction(pSubnet);
    pOptionsMenu->addAction(pAnnotation);
    pOptionsMenu->addAction(pWrangle);

    connect(pNode, &QAction::triggered, this, [=](bool bChecked) {
        if (bChecked)
            m_searchOpts |= SEARCH_NODECLS;
        else
            m_searchOpts &= (~(int)SEARCH_NODECLS);
    });

    connect(pNodeID, &QAction::triggered, this, [=](bool bChecked) {
        if (bChecked)
        m_searchOpts |= SEARCH_NODEID;
        else
            m_searchOpts &= (~(int)SEARCH_NODEID);
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
            m_searchOpts |= SEARCH_ARGS;
        else
            m_searchOpts &= (~(int)SEARCH_ARGS);
        });

    pOptionsMenu->exec(QCursor::pos());
    pOptionsMenu->deleteLater();
}

void ZenoGraphsEditor::onNewSubgraph()
{
    bool bOk = false;
    QString newSubgName = QInputDialog::getText(this, tr("create subnet"), tr("new subgraph name:")
        , QLineEdit::Normal, "SubgraphName", &bOk);

    if (newSubgName.compare("main", Qt::CaseInsensitive) == 0)
    {
        QMessageBox msg(QMessageBox::Warning, tr("Zeno"), tr("main graph is not allowed to be created"));
        msg.exec();
        return;
    }

    if (bOk) {
        m_pSubgraphs->newSubgraph(newSubgName);
    }
}

void ZenoGraphsEditor::onSubnetOptionClicked()
{
    QMenu* pOptionsMenu = new QMenu;

    QAction* pNewSubg = new QAction(tr("create subnet"));
    QAction* pSubnetMap = new QAction(tr("subnet map"));
    QAction* pImpFromFile = new QAction(tr("import from local file"));
    QAction* pImpFromSys = new QAction(tr("import system subnet"));

    pOptionsMenu->addAction(pNewSubg);
    pOptionsMenu->addAction(pSubnetMap);
    pOptionsMenu->addSeparator();
    pOptionsMenu->addAction(pImpFromFile);
    pOptionsMenu->addAction(pImpFromSys);

    connect(pNewSubg, &QAction::triggered, this, &ZenoGraphsEditor::onNewSubgraph);
    connect(pSubnetMap, &QAction::triggered, this, [=]() {

    });
    connect(pImpFromFile, &QAction::triggered, this, [=]() {
        m_mainWin->importSubGraph();
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
    /*
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
    */

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
        /*
        case Side_Subnet: m_ui->subnetBtn->setChecked(false); break;
        case Side_Tree: m_ui->treeviewBtn->setChecked(false); break;
        case Side_Search: m_ui->searchBtn->setChecked(false); break;
        */
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
                //m_ui->subnetBtn->setChecked(true);
                m_ui->stackedWidget->setCurrentWidget(m_ui->subnetPage);
                break;
            }
            case Side_Tree:
            {
                //m_ui->treeviewBtn->setChecked(true);
                m_ui->stackedWidget->setCurrentWidget(m_ui->treePage);
                break;
            }
            case Side_Search:
            {
                //m_ui->searchBtn->setChecked(true);
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
    activateTab(index.data(ROLE_OBJNAME).toString());
}

void ZenoGraphsEditor::selectTab(const QString& subGraphName, const QString& path, std::vector<QString> & objIds)
{
    auto graphsMgm = zenoApp->graphsManagment();
    IGraphsModel* pModel = graphsMgm->currentModel();

    if (!pModel || !pModel->index(subGraphName).isValid())
        return;

    int idx = tabIndexOfName(subGraphName);
    if (idx == -1)
    {
        const QModelIndex& subgIdx = pModel->index(subGraphName);

        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(graphsMgm->gvScene(subgIdx));
        if (!pScene)
        {
            pScene = new ZenoSubGraphScene(graphsMgm);
            graphsMgm->addScene(subgIdx, pScene);
            pScene->initModel(m_pSubgraphs, subgIdx);
        }

        pScene->select(objIds);
    }
    const QModelIndex& subgIdx = pModel->index(subGraphName);
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(graphsMgm->gvScene(subgIdx));
    if (!pScene)
    {
        pScene = new ZenoSubGraphScene(graphsMgm);
        graphsMgm->addScene(subgIdx, pScene);
        pScene->initModel(m_pSubgraphs, subgIdx);
    }

    pScene->select(objIds);
}

void ZenoGraphsEditor::activateTabOfTree(const QString &path, const QString &nodeid, bool isError) {
    auto graphsMgm = zenoApp->graphsManagment();
    IGraphsModel *pNodeModel = graphsMgm->currentModel();
    ZASSERT_EXIT(pNodeModel);
    QModelIndex subgIdx = pNodeModel->indexFromPath(path);

    int idx = tabIndexOfName(path);
    if (idx == -1)
    {
        ZenoSubGraphScene *pScene = qobject_cast<ZenoSubGraphScene *>(graphsMgm->gvScene(subgIdx));
        if (!pScene) {
            pScene = new ZenoSubGraphScene(graphsMgm);
            graphsMgm->addScene(subgIdx, pScene);
            pScene->initModel(pNodeModel, subgIdx);
        }

        ZenoSubGraphView *pView = new ZenoSubGraphView;
        connect(pView, &ZenoSubGraphView::zoomed, pScene, &ZenoSubGraphScene::onZoomed);
        connect(pView, &ZenoSubGraphView::zoomed, this, &ZenoGraphsEditor::zoomed);
        pView->initScene(pScene);

        idx = m_ui->graphsViewTab->addTab(pView, path);

        QString tabIcon;
        if (path.compare("/main", Qt::CaseInsensitive) == 0)
            tabIcon = ":/icons/subnet-main.svg";
        else
            tabIcon = ":/icons/subnet.svg";
        m_ui->graphsViewTab->setTabIcon(idx, QIcon(tabIcon));
        connect(pView, &ZenoSubGraphView::pathUpdated, this, [=](QString newPath) {
            activateTabOfTree(newPath, "", false);
        });
    }
    m_ui->graphsViewTab->setCurrentIndex(idx);

    ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
    ZASSERT_EXIT(pView);

    pView->resetPath(pNodeModel, path, subgIdx, nodeid, isError);
    m_mainWin->onNodesSelected(subgIdx, pView->scene()->selectNodesIndice(), true);
}

void ZenoGraphsEditor::activateTab(const QString& subGraphName, const QString& path, const QString& objId, bool isError)
{
    auto graphsMgm = zenoApp->graphsManagment();
    IGraphsModel* pSubgraphs = graphsMgm->sharedSubgraphs();
    ZASSERT_EXIT(pSubgraphs);
    if (!pSubgraphs->index(subGraphName).isValid())
        return;

    int idx = tabIndexOfName(subGraphName);
    const QModelIndex& subgIdx = pSubgraphs->index(subGraphName);
    if (idx == -1)
    {
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(graphsMgm->gvScene(subgIdx));
        if (!pScene)
        {
            pScene = new ZenoSubGraphScene(graphsMgm);
            graphsMgm->addScene(subgIdx, pScene);
            pScene->initModel(pSubgraphs, subgIdx);
        }

        ZenoSubGraphView* pView = new ZenoSubGraphView;
        connect(pView, &ZenoSubGraphView::zoomed, pScene, &ZenoSubGraphScene::onZoomed);
        connect(pView, &ZenoSubGraphView::zoomed, this, &ZenoGraphsEditor::zoomed);
        pView->initScene(pScene);

        idx = m_ui->graphsViewTab->addTab(pView, subGraphName);

        QString tabIcon;
        if (subGraphName.compare("main", Qt::CaseInsensitive) == 0)
            tabIcon = ":/icons/subnet-main.svg";
        else
            tabIcon = ":/icons/subnet-general.svg";
        m_ui->graphsViewTab->setTabIcon(idx, QIcon(tabIcon));

        connect(pView, &ZenoSubGraphView::pathUpdated, this, [=](QString newPath) {
            QStringList L = newPath.split("/", QtSkipEmptyParts);
            QString subgName = L.last();
            activateTab(subgName, newPath);
        });
    }
    m_ui->graphsViewTab->setCurrentIndex(idx);

    ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
    ZASSERT_EXIT(pView);
    pView->resetPath(pSubgraphs, path, subgIdx, objId, isError);

    m_mainWin->onNodesSelected(pSubgraphs->index(subGraphName), pView->scene()->selectNodesIndice(), true);
}

void ZenoGraphsEditor::showFloatPanel(const QModelIndex &subgIdx, const QModelIndexList &nodes) {
    ZenoSubGraphView *pView = qobject_cast<ZenoSubGraphView *>(m_ui->graphsViewTab->currentWidget());
    if (pView != NULL)
    {
        pView->showFloatPanel(subgIdx, nodes);
    }
}

void ZenoGraphsEditor::onTreeItemActivated(const QModelIndex& index)
{
    QModelIndex idx = index;
    QModelIndex subgIdx = idx.parent();
    if (subgIdx.isValid())
    {
        const QString& subgPath = subgIdx.data(ROLE_OBJPATH).toString();
        const QString& nodeid = idx.data(ROLE_OBJID).toString();
        activateTabOfTree(subgPath, nodeid);
        return;
    }
}

void ZenoGraphsEditor::onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx)
{
    if (UiHelper::getGraphsBySubg(subgIdx) == m_pNodeModel)
    {
        const QString& subgPath = nodeIdx.data(ROLE_OBJPATH).toString();
        activateTabOfTree(subgPath);
    }
    else
    {
        activateTab(nodeIdx.data(ROLE_OBJNAME).toString());
    }
}

void ZenoGraphsEditor::onLogInserted(const QModelIndex& parent, int first, int last)
{
    if (!m_pNodeModel)
        return;
    QStandardItemModel* logModel = qobject_cast<QStandardItemModel*>(sender());
    const QModelIndex& idx = logModel->index(first, 0, parent);
    if (idx.isValid())
    {
        QString objId = idx.data(ROLE_NODE_IDENT).toString();
        QtMsgType type = (QtMsgType)idx.data(ROLE_LOGTYPE).toInt();
        if (!objId.isEmpty() && type == QtFatalMsg)
        {
            if (objId.indexOf('/') != -1)
            {
                auto lst = objId.split('/', QtSkipEmptyParts);
                objId = lst.last();
            }

            QList<SEARCH_RESULT> results = m_pNodeModel->search(objId, SEARCH_NODEID, SEARCH_MATCH_EXACTLY);
            for (int i = 0; i < results.length(); i++)
            {
                const SEARCH_RESULT& res = results[i];
                const QString &subgPath = res.subgIdx.data(ROLE_OBJPATH).toString();
                const QString &subgId = res.targetIdx.data(ROLE_OBJID).toString();

                QVariant varFocusOnError = ZenoSettingsManager::GetInstance().getValue(zsTraceErrorNode);

                bool bFocusOnError = true;
                if (varFocusOnError.type() == QVariant::Bool) {
                    bFocusOnError = varFocusOnError.toBool();
                }
                if (bFocusOnError)
                {
                    activateTabOfTree(subgPath, subgId, true);
                    if (i == results.length() - 1)
                        break;

                    QMessageBox msgbox(QMessageBox::Question, "", tr("next one?"), QMessageBox::Yes | QMessageBox::No);
                    int ret = msgbox.exec();
                    if (ret & QMessageBox::Yes) {
                    }
                    else {
                        break;
                    }
                }
                else
                {
                    const QModelIndex &subgIdx = res.subgIdx;
                    auto graphsMgm = zenoApp->graphsManagment();
                    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(graphsMgm->gvScene(subgIdx));
                    if (!pScene) {
                        pScene = new ZenoSubGraphScene(graphsMgm);
                        graphsMgm->addScene(subgIdx, pScene);
                        pScene->initModel(m_pNodeModel, subgIdx);
                    }
                    pScene->markError(objId);
                }
            }
        }
    }
}

void ZenoGraphsEditor::onSearchEdited(const QString& content)
{
    QList<SEARCH_RESULT> results = m_pNodeModel->search(content, m_searchOpts, SEARCH_FUZZ);
    QList<SEARCH_RESULT> subgResults;
    if (m_pSubgraphs)
    {
        subgResults = m_pSubgraphs->search(content, m_searchOpts, SEARCH_FUZZ);
        results.append(subgResults);
    }

    QStandardItemModel* pModel = new QStandardItemModel(this);

    for (SEARCH_RESULT res : results)
    {
        if (res.type == SEARCH_SUBNET)
        {
            QString ident;
            if (res.subgIdx.data(ROLE_OBJID).isValid())
                ident = res.subgIdx.data(ROLE_OBJID).toString();
            else
                ident = res.subgIdx.data(ROLE_OBJNAME).toString();
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJID, ident, 1, Qt::MatchExactly);
            if (lst.size() == 0)
            {
                //add subnet
                QStandardItem* pItem = new QStandardItem(ident + " (Subnet)");
                pItem->setData(ident, ROLE_OBJID);
                pItem->setData(res.subgIdx, ROLE_SUBGRAPH_IDX);
                pModel->appendRow(pItem);
            }
        }
        else if (res.type == SEARCH_NODECLS || res.type == SEARCH_NODEID || res.type == SEARCH_ARGS || res.type == SEARCH_CUSTOM_NAME)
        {
            QString ident;
            if (res.subgIdx.data(ROLE_OBJID).isValid())
                ident = res.subgIdx.data(ROLE_OBJID).toString();
            else
                ident = res.subgIdx.data(ROLE_OBJNAME).toString();
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJID, ident, 1, Qt::MatchExactly);

            QStandardItem* parentItem = nullptr;
            if (lst.size() == 0)
            {
                //add subnet
                parentItem = new QStandardItem(ident + " (Subnet)");
                parentItem->setData(ident, ROLE_OBJID);
                QVariant parentIdx = res.subgIdx.data(ROLE_SUBGRAPH_IDX);
                parentItem->setData(parentIdx.isValid() ? parentIdx : res.subgIdx, ROLE_SUBGRAPH_IDX);
                pModel->appendRow(parentItem);
            }
            else
            {
                ZASSERT_EXIT(lst.size() == 1);
                parentItem = pModel->itemFromIndex(lst[0]);
            }

            QString nodeIdent = res.targetIdx.data(ROLE_OBJID).toString();
            QStandardItem* pItem = new QStandardItem(nodeIdent);
            pItem->setData(res.targetIdx.data(ROLE_OBJID).toString(), ROLE_OBJID);
            pItem->setData(res.subgIdx, ROLE_SUBGRAPH_IDX);
            parentItem->appendRow(pItem);
        }
    }

    if (QAbstractItemModel* model = m_ui->searchResView->model())
    {
        delete model;
        model = nullptr;
    }
    if (QAbstractItemDelegate* pDelegate = m_ui->searchResView->itemDelegate())
    {
        delete pDelegate;
        pDelegate = nullptr;
    }
    m_ui->searchResView->setModel(pModel);
    m_ui->searchResView->setItemDelegate(new SearchItemDelegate(content));
    m_ui->searchResView->expandAll();
}

void ZenoGraphsEditor::onSearchItemClicked(const QModelIndex& index)
{
    QString objId = index.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = index.data(ROLE_SUBGRAPH_IDX).value<QModelIndex>();
    QString subgName = subgIdx.data(ROLE_OBJNAME).toString();
    if (UiHelper::getGraphsBySubg(subgIdx) == m_pNodeModel)
    {
        QString subgPath = subgIdx.data(ROLE_OBJPATH).toString();
        activateTabOfTree(subgPath, objId == "main" ? "" : objId);
    }
    else
    {
        activateTab(subgIdx.data(ROLE_OBJNAME).toString(), "", index.parent().isValid() ? objId : "");
    }
}

void ZenoGraphsEditor::toggleViewForSelected(bool bOn)
{
    ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
    if (pView)
    {
        ZenoSubGraphScene* pScene = pView->scene();
        QModelIndexList nodes = pScene->selectNodesIndice();
        const QModelIndex& subgIdx = pScene->subGraphIndex();
        for (const QModelIndex& idx : nodes)
        {
            STATUS_UPDATE_INFO info;
            int options = idx.data(ROLE_OPTIONS).toInt();
            info.oldValue = options;
            if (bOn) {
                options |= OPT_VIEW;
            }
            else {
                options &= (~OPT_VIEW);
            }
            info.role = ROLE_OPTIONS;
            info.newValue = options;
            IGraphsModel *pModel = UiHelper::getGraphsBySubg(subgIdx);
            ZASSERT_EXIT(pModel);
            pModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, subgIdx);
        }
    }
}

void ZenoGraphsEditor::onSubnetListPanel(bool bShow, SideBarItem item) 
{
    QModelIndex idx = m_sideBarModel->match(m_sideBarModel->index(0, 0), Qt::UserRole + 1, item)[0];
    m_selection->setCurrentIndex(idx, QItemSelectionModel::SelectCurrent);
    m_ui->stackedWidget->setVisible(bShow);
}

void ZenoGraphsEditor::onMenuActionTriggered(QAction* pAction)
{
    onAction(pAction);
}

void ZenoGraphsEditor::onCommandDispatched(QAction* pAction, bool bTriggered)
{
    onAction(pAction);
}

void ZenoGraphsEditor::onTreeItemSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected) 
{
    QModelIndexList lst = m_ui->subnetTree->selectionModel()->selectedIndexes();
    if (lst.isEmpty())
        return;

    QSortFilterProxyModel* pProxyModel = qobject_cast<QSortFilterProxyModel*>(m_ui->subnetTree->model());
    QModelIndex idx = pProxyModel->mapToSource(lst.first());
    if (lst.size() == 1) 
    {
        onTreeItemActivated(idx);
    } 
    else 
    {
        QModelIndexList indexs;
        if (!idx.parent().isValid())
            return;
        QString parentName = idx.parent().data(ROLE_OBJNAME).toString();
        indexs << idx;
        for (const auto& index : lst) 
        {
            if (index == idx)
                continue;
            if (index.parent().data(ROLE_OBJNAME).toString() == parentName) {
                indexs << index;
            }
        }
        ZenoSubGraphView *pView = qobject_cast<ZenoSubGraphView *>(m_ui->graphsViewTab->currentWidget());
        ZASSERT_EXIT(pView);
        pView->selectNodes(indexs);
    }
}

void ZenoGraphsEditor::onAction(QAction* pAction, const QVariantList& args, bool bChecked)
{
    int actionType = pAction->property("ActionType").toInt();
    if (actionType == ZenoMainWindow::ACTION_COLLASPE)
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        ZASSERT_EXIT(pView);
        QModelIndex subgIdx = pView->scene()->subGraphIndex();
        IGraphsModel *pModel = UiHelper::getGraphsBySubg(subgIdx);
        ZASSERT_EXIT(pModel);
        pModel->collaspe(subgIdx);
    }
    else if (actionType == ZenoMainWindow::ACTION_EXPAND) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        ZASSERT_EXIT(pView);
        QModelIndex subgIdx = pView->scene()->subGraphIndex();
        IGraphsModel* pModel = UiHelper::getGraphsBySubg(subgIdx);
        ZASSERT_EXIT(pModel);
        pModel->expand(subgIdx);
    }
    else if (actionType == ZenoMainWindow::ACTION_OPEN_VIEW) 
    {
        toggleViewForSelected(true);
    }
    else if (actionType == ZenoMainWindow::ACTION_CLEAR_VIEW) 
    {
        toggleViewForSelected(false);
    }    
    else if (actionType == ZenoMainWindow::ACTION_CUSTOM_UI) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        if (pView)
        {
            ZenoSubGraphScene* pScene = pView->scene();
            QModelIndexList nodes = pScene->selectNodesIndice();
            if (nodes.size() == 1)
            {
                QModelIndex nodeIdx = nodes[0];
                //only subgraph node
                IGraphsModel* pModel = UiHelper::getGraphsBySubg(pScene->subGraphIndex());
                ZASSERT_EXIT(pModel);
                if (!pModel->IsSubGraphNode(nodeIdx))
                {
                    QMessageBox::information(this, tr("Info"), tr("Cannot edit parameters!"));
                    return;
                }
                QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
                ZASSERT_EXIT(viewParams);
                ZEditParamLayoutDlg dlg(viewParams, true, nodeIdx, pModel, this);
                dlg.exec();
            }
        }
    } 
    else if (actionType == ZenoMainWindow::ACTION_GROUP) 
    {
        ZenoSubGraphView *pView = qobject_cast<ZenoSubGraphView *>(m_ui->graphsViewTab->currentWidget());
        if (pView) 
        {
            ZenoSubGraphScene *pScene = pView->scene();
            QModelIndex subgIdx = pScene->subGraphIndex();
            IGraphsModel* pModel = UiHelper::getGraphsBySubg(subgIdx);
            ZASSERT_EXIT(pModel);
            NodesMgr::createNewNode(pModel, subgIdx, "Group", QPointF());
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_EASY_GRAPH) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        if (pView)
        {
            ZenoSubGraphScene* pScene = pView->scene();
            QModelIndexList nodes = pScene->selectNodesIndice();
            QModelIndexList links = pScene->selectLinkIndice();
            bool bOk = false;
            QString newSubgName = QInputDialog::getText(this, tr("create subnet"), tr("new subgraph name:") , QLineEdit::Normal, "subgraph name", &bOk);
            if (bOk)
            {
                QModelIndex fromSubgIdx = pView->scene()->subGraphIndex();
                IGraphsModel *pModel = UiHelper::getGraphsBySubg(fromSubgIdx);
                ZASSERT_EXIT(pModel);
                QModelIndex toSubgIdx = pModel->extractSubGraph(nodes, links, fromSubgIdx, newSubgName, true);
                if (toSubgIdx.isValid())
                {
                    activateTab(toSubgIdx.data(ROLE_OBJNAME).toString());
                }
            }
            else
            {
                //todo: msg to feedback.
            }
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_SET_NASLOC) 
    {
        QSettings settings(zsCompanyName, zsEditor);
        QString v = settings.value("nas_loc").toString();

        QDialog dlg(this);
        QGridLayout *pLayout = new QGridLayout(&dlg);
        QDialogButtonBox *pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        ZPathEdit *pathLineEdit = new ZPathEdit(v, &dlg);
        pLayout->addWidget(new QLabel("Set NASLOC"), 2, 0);
        pLayout->addWidget(pathLineEdit, 2, 1);
        pLayout->addWidget(pButtonBox, 4, 1);
        connect(pButtonBox, SIGNAL(accepted()), &dlg, SLOT(accept()));
        connect(pButtonBox, SIGNAL(rejected()), &dlg, SLOT(reject()));
        if (QDialog::Accepted == dlg.exec()) {
            QString text = pathLineEdit->text();
            text.replace('\\', '/');
            settings.setValue("nas_loc", text);
            // refresh settings (zeno::setConfigVariable), only needed in single-process mode
            startUp();
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_ZENCACHE) 
    {
        QSettings settings(zsCompanyName, zsEditor);

        auto &inst = ZenoSettingsManager::GetInstance();

        QVariant varEnableCache = inst.getValue("zencache-enable");
        QVariant varAutoRemove = inst.getValue("zencache-autoremove");
        QVariant varCacheRoot = inst.getValue("zencachedir");
        QVariant varCacheNum = inst.getValue("zencachenum");

        bool bEnableCache = varEnableCache.isValid() ? varEnableCache.toBool() : false;
        bool bAutoRemove = varAutoRemove.isValid() ? varAutoRemove.toBool() : false;
        QString cacheRootDir = varCacheRoot.isValid() ? varCacheRoot.toString() : "";
        int cacheNum = varCacheNum.isValid() ? varCacheNum.toInt() : 1;

        ZPathEdit *pathLineEdit = new ZPathEdit(cacheRootDir);
        pathLineEdit->setFixedWidth(256);
        pathLineEdit->setEnabled(!bAutoRemove && bEnableCache);
        QCheckBox *pAutoDelCache = new QCheckBox;
        pAutoDelCache->setCheckState(bAutoRemove ? Qt::Checked : Qt::Unchecked);
        pAutoDelCache->setEnabled(bEnableCache);
        connect(pAutoDelCache, &QCheckBox::stateChanged, [=](bool state) {
            pathLineEdit->setText("");
            pathLineEdit->setEnabled(!state);
        });

        QSpinBox* pSpinBox = new QSpinBox;
        pSpinBox->setRange(1, 10000);
        pSpinBox->setValue(cacheNum);
        pSpinBox->setEnabled(bEnableCache);

        QCheckBox *pCheckbox = new QCheckBox;
        pCheckbox->setCheckState(bEnableCache ? Qt::Checked : Qt::Unchecked);
        connect(pCheckbox, &QCheckBox::stateChanged, [=](bool state) {
            if (!state)
            {
                pSpinBox->clear();
                pathLineEdit->clear();
                pAutoDelCache->setCheckState(Qt::Unchecked);
            }
            pSpinBox->setEnabled(state);
            pathLineEdit->setEnabled(state);
            pAutoDelCache->setEnabled(state);
        });

        QDialogButtonBox* pButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

        QDialog dlg(this);
        QGridLayout* pLayout = new QGridLayout;
        pLayout->addWidget(new QLabel("enable cache"), 0, 0);
        pLayout->addWidget(pCheckbox, 0, 1);
        pLayout->addWidget(new QLabel("cache num"), 1, 0);
        pLayout->addWidget(pSpinBox, 1, 1);
        pLayout->addWidget(new QLabel("cache root"), 2, 0);
        pLayout->addWidget(pathLineEdit, 2, 1);
        pLayout->addWidget(new QLabel("temp cache directory"), 3, 0);
        pLayout->addWidget(pAutoDelCache, 3, 1);
        pLayout->addWidget(pButtonBox, 4, 1);

        connect(pButtonBox, SIGNAL(accepted()), &dlg, SLOT(accept()));
        connect(pButtonBox, SIGNAL(rejected()), &dlg, SLOT(reject()));

        dlg.setLayout(pLayout);
        if (QDialog::Accepted == dlg.exec())
        {
            inst.setValue("zencache-enable", pCheckbox->checkState() == Qt::Checked);
            inst.setValue("zencache-autoremove", pAutoDelCache->checkState() == Qt::Checked);
            inst.setValue("zencachedir", pathLineEdit->text());
            inst.setValue("zencachenum", pSpinBox->value());
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_ZOOM) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        if (pView)
        {
            if (!args.isEmpty() && (args[0].type() == QMetaType::Float || args[0].type() == QMetaType::Double)) {
                pView->setZoom(args[0].toFloat());
            }
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_UNDO) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        ZASSERT_EXIT(pView);
        QModelIndex subgIdx = pView->scene()->subGraphIndex();
        IGraphsModel* pGraphsModel = UiHelper::getGraphsBySubg(subgIdx);
        ZASSERT_EXIT(pGraphsModel);
        pGraphsModel->undo();
    }
    else if (actionType == ZenoMainWindow::ACTION_REDO) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        ZASSERT_EXIT(pView);
        QModelIndex subgIdx = pView->scene()->subGraphIndex();
        IGraphsModel* pGraphsModel = UiHelper::getGraphsBySubg(subgIdx);
        ZASSERT_EXIT(pGraphsModel);
        pGraphsModel->redo();
    }
    else if (actionType == ZenoMainWindow::ACTION_SELECT_NODE) 
    {
        ZenoSubGraphView* pView = qobject_cast<ZenoSubGraphView*>(m_ui->graphsViewTab->currentWidget());
        QModelIndex nodeIdx = pAction->data().toModelIndex();
        if (pView && nodeIdx.isValid())
            pView->focusOn(nodeIdx.data(ROLE_OBJID).toString());
    }
    else if (actionType == ZenoMainWindow::ACTION_NEW_SUBGRAPH)
    {
        onNewSubgraph();
    }
}
