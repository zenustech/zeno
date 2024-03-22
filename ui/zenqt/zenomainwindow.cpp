#include "zenomainwindow.h"
#include "layout/zdockwidget.h"
#include "model/graphsmanager.h"
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/GenericObject.h>
#include "nodeeditor/gv/zenographseditor.h"
#include "layout/zdocktabwidget.h"
#include "layout/docktabcontent.h"
#include "panel/zenodatapanel.h"
#include "panel/zenoproppanel.h"
#include "panel/zenospreadsheet.h"
#include "panel/zlogpanel.h"
#include "widgets/ztimeline.h"
#include "widgets/ztoolbar.h"
#include "viewport/viewportwidget.h"
#include "viewport/optixviewport.h"
#include "viewport/zoptixviewport.h"
#include "viewport/zenovis.h"
#include "zenoapplication.h"
#include <zeno/utils/log.h>
#include <zeno/utils/envconfig.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/io/zsg2reader.h>
#include <zeno/io/zenwriter.h>
#include <zenovis/DrawOptions.h>
#include "uicommon.h"
#include "style/zenostyle.h"
#include "util/uihelper.h"
#include "util/log.h"
#include "dialog/zfeedbackdlg.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"
#include "panel/zenolights.h"
#include "nodeeditor/gv/zenosubgraphscene.h"
#include "viewport/recordvideomgr.h"
#include "viewport/displaywidget.h"
#include "viewport/optixviewport.h"
#include "ui_zenomainwindow.h"
#include <QJsonDocument>
#include "dialog/zdocklayoutmangedlg.h"
#include "panel/zenoimagepanel.h"
#include "dialog/zshortcutsettingdlg.h"
#include "settings/zenosettingsmanager.h"
#include "util/apphelper.h"
#include "dialog/zaboutdlg.h"
#include <zeno/extra/GlobalError.h>
#include <zeno/extra/GlobalState.h>
#include "dialog/ZImportSubgraphsDlg.h"
#include "dialog/zcheckupdatedlg.h"
#include "dialog/zrestartdlg.h"
#include "dialog/zpreferencesdlg.h"
#include "model/GraphsTreeModel.h"
#include <zeno/core/Session.h>
#include <zeno/utils/api.h>
#include "calculation/calculationmgr.h"


const QString g_latest_layout = "LatestLayout";

ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags, PANEL_TYPE onlyView)
    : QMainWindow(parent, flags)
    , m_bInDlgEventloop(false)
    , m_bAlways(false)
    , m_bAlwaysLightCamera(false)
    , m_bAlwaysMaterial(false)
    , m_pTimeline(nullptr)
    , m_layoutRoot(nullptr)
    , m_nResizeTimes(0)
    , m_bOnlyOptix(false)
{
    init(onlyView);
    setContextMenuPolicy(Qt::NoContextMenu);
    setFocusPolicy(Qt::ClickFocus);

//#ifdef __linux__
    if (char *p = zeno::envconfig::get("OPEN")) {
        zeno::log_info("ZENO_OPEN: {}", p);
        openFile(p);
    }
//#endif
}

ZenoMainWindow::~ZenoMainWindow()
{
}

void ZenoMainWindow::init(PANEL_TYPE onlyView)
{
    m_ui = new Ui::MainWindow;
    m_ui->setupUi(this);

    initMenu();
    initLive();
    initDocks(onlyView);
    initWindowProperty();

    addToolBar(Qt::LeftToolBarArea, new FakeToolbar(false));
    addToolBar(Qt::RightToolBarArea, new FakeToolbar(false));
    addToolBar(Qt::BottomToolBarArea, new FakeToolbar(true));
    addToolBar(Qt::TopToolBarArea, new FakeToolbar(true));

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(11, 11, 11));
    setAutoFillBackground(true);
    setPalette(pal);
    setAcceptDrops(true);

    m_ui->statusbar->showMessage(tr("Status Bar"));
    connect(this, &ZenoMainWindow::recentFilesChanged, this, [=](const QObject *sender) {
        if (sender != this)
            loadRecentFiles();
    });

    auto calcMgr = zenoApp->calculationMgr();
    if (calcMgr)
        connect(calcMgr, &CalculationMgr::calcFinished, this, &ZenoMainWindow::onCalcFinished);
}

void ZenoMainWindow::initWindowProperty()
{
    auto pGraphsMgm = zenoApp->graphsManager();
    setWindowTitle(AppHelper::nativeWindowTitle(""));
    connect(pGraphsMgm, &GraphsManager::fileOpened, this, [=](QString fn) {
        QFileInfo info(fn);
        QString path = info.filePath();
        QString title = AppHelper::nativeWindowTitle(path);
        updateNativeWinTitle(title);
    });
    connect(pGraphsMgm, &GraphsManager::modelInited, this, [=]() {
        //new file
        QString title = AppHelper::nativeWindowTitle(tr("new file"));
        updateNativeWinTitle(title);
    });
    connect(pGraphsMgm, &GraphsManager::fileClosed, this, [=]() { 
        QString title = AppHelper::nativeWindowTitle("");
        updateNativeWinTitle(title);
    });
    connect(pGraphsMgm, &GraphsManager::fileSaved, this, [=](QString path) {
        QString title = AppHelper::nativeWindowTitle(path);
        updateNativeWinTitle(title);
    });
    connect(this, &ZenoMainWindow::visFrameUpdated, this, &ZenoMainWindow::onZenovisFrameUpdate);
}

void ZenoMainWindow::updateNativeWinTitle(const QString& title)
{
    QWidgetList lst = QApplication::topLevelWidgets();
    for (auto wid : lst)
    {
        if (qobject_cast<ZDockWidget*>(wid) ||
            qobject_cast<ZenoMainWindow*>(wid))
        {
            wid->setWindowTitle(title);
        }
    }
}

void ZenoMainWindow::initLive() {

}

void ZenoMainWindow::initMenu()
{
    //to merge:
/*
        QAction *pAction = new QAction(tr("New"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+N")));
        pAction->setShortcutContext(Qt::ApplicationShortcut);
        //QMenu *pNewMenu = new QMenu;
        //QAction *pNewGraph = pNewMenu->addAction("New Scene");
        connect(pAction, SIGNAL(triggered()), this, SLOT(onNewFile()));
 */
    setActionProperty();

    QSettings settings(zsCompanyName, zsEditor);
    QVariant use_chinese = settings.value("use_chinese");
    m_ui->actionEnglish_Chinese->setChecked(use_chinese.isNull() || use_chinese.toBool());

    auto actions = findChildren<QAction*>(QString(), Qt::FindDirectChildrenOnly);
    for (QAction* action : actions)
    {
        connect(action, SIGNAL(triggered(bool)), this, SLOT(onMenuActionTriggered(bool)));  
        setActionIcon(action);
    }

    m_ui->menubar->setProperty("cssClass", "mainWin");
    //qt bug: qss font is not valid on menubar.
    QFont font = QApplication::font();
    font.setPointSize(10);
    font.setWeight(QFont::Medium);
    m_ui->menubar->setFont(font);

    //default layout
    QJsonObject obj = readDefaultLayout();
    QStringList lst = obj.keys();
    initCustomLayoutAction(lst, true);
    //check user saved layout.
    loadSavedLayout();
    //init recent files
    loadRecentFiles();
    initShortCut();
}

void ZenoMainWindow::onMenuActionTriggered(bool bTriggered)
{
    QAction* pAction = qobject_cast<QAction*>(sender());
    QVariant var = pAction->property("ActionType");
    int actionType = -1;
    if (var.type() == QVariant::Int)
        actionType = pAction->property("ActionType").toInt();
    setActionIcon(pAction);
    switch (actionType)
    {
    case ACTION_NEWFILE: {
        onNewFile();
        break;
    }
    case ACTION_OPEN: {
        openFileDialog();
        break;
    }
    case ACTION_SAVE: {
        save();
        break;
    }
    case ACTION_SAVE_AS: {
        saveAs();
        break;
    }
    case ACTION_IMPORT: {
        importGraph();
        break;
    }
    case ACTION_EXPORT_GRAPH: {
        exportGraph();
        break;
    }
    case ACTION_CLOSE: {
        saveQuitShowWelcom();
        break;
    }
    case ACTION_SAVE_LAYOUT: {
        saveDockLayout();
        break;
    }
    case ACTION_LAYOUT_MANAGE: {
        manageCustomLayout();
        break;
    }
    case ACTION_LANGUAGE: {
        onLangChanged(bTriggered);
        break;
    }
    case ACTION_SHORTCUTLIST: {
        QDesktopServices::openUrl(QUrl("http://doc.zenustech.com/project-3/doc-135/"));
        break;
    }
    case ACTION_SCREEN_SHOOT: {
        screenShoot();
        break;
    }
    case ACTION_SET_SHORTCUT: {
        shortCutDlg();
        break;
    }
    case ACTION_PREFERENCES: {
        ZPreferencesDlg dlg;
        dlg.exec();
        break;
    }
    case ACTION_ABOUT: {
        ZAboutDlg dlg(this);
        dlg.exec();
        break;
    }
    case ACTION_FEEDBACK: {
        onFeedBack();
        break;
    }
    case ACTION_CHECKUPDATE: {
        onCheckUpdate();
        break;
    }
    default: {
        dispatchCommand(pAction, bTriggered);
        break;
    }
    }
}

void ZenoMainWindow::dispatchCommand(QAction* pAction, bool bTriggered)
{
    if (!pAction)
        return;

    //dispatch to every panel.
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    DisplayWidget* pViewport = nullptr;
    ZenoGraphsEditor* pEditor = nullptr;
    for (ZDockWidget* pDock : docks)
    {
        if (!pViewport)
            pViewport = pDock->getUniqueViewport();
        if (!pEditor)
            pEditor = pDock->getAnyEditor();
        for (int i = 0; i < pDock->count(); i++)
        {
            DisplayWidget* pDisplay = qobject_cast<DisplayWidget*>(pDock->widget(i));
            if (pDisplay)
            {
                int actionType = pAction->property("ActionType").toInt();
                //pDisplay->onCommandDispatched(actionType, bTriggered);
            }
        }
    }
    if (pEditor)
    {
        pEditor->onCommandDispatched(pAction, bTriggered);
    }
    if (pViewport)
    {
        int actionType = pAction->property("ActionType").toInt();
        pViewport->onCommandDispatched(actionType, bTriggered);
    }
}

void ZenoMainWindow::loadSavedLayout()
{
    m_ui->menuCustom_Layout->clear();
    //custom layout
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    QStringList lst = settings.childGroups();
    settings.endGroup();
    if (!lst.isEmpty()) {
        initCustomLayoutAction(lst, false);
    }
}

void ZenoMainWindow::saveDockLayout()
{
    bool bOk = false;
    QString name = QInputDialog::getText(this, tr("Save Layout"), tr("layout name:"),
        QLineEdit::Normal, "layout_1", &bOk);
    if (bOk)
    {
        QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
        settings.beginGroup("layout");
        if (settings.childGroups().indexOf(name) != -1)
        {
            QMessageBox msg(QMessageBox::Question, "", tr("alreday has same layout, override?"),
                QMessageBox::Ok | QMessageBox::Cancel);
            int ret = msg.exec();
            if (ret == QMessageBox::Cancel)
            {
                settings.endGroup();
                return;
            }
        }

        QString layoutInfo = exportLayout(m_layoutRoot, size());
        settings.beginGroup(name);
        settings.setValue("content", layoutInfo);
        settings.endGroup();
        settings.endGroup();
        loadSavedLayout();
    }
}

void ZenoMainWindow::saveLayout2()
{
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "JSON file(*.json);;");
    writeLayout(m_layoutRoot, size(), path);
}

void ZenoMainWindow::onLangChanged(bool bChecked)
{
    QSettings settings(zsCompanyName, zsEditor);
    settings.setValue("use_chinese", bChecked);
    QMessageBox msg(QMessageBox::Information, tr("Language"),
        tr("Please restart Zeno to apply changes."),
        QMessageBox::Ok, this);
    msg.exec();
}

void ZenoMainWindow::resetDocks(PtrLayoutNode root)
{
    if (root == nullptr)
        return;

    m_layoutRoot.reset();
    auto docks = findChildren<ZDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget *pDock : docks) {
        pDock->close();
        //pDock->testCleanupGL();
        //delete pDock;
    }

    m_layoutRoot = root;
    ZDockWidget* cake = new ZDockWidget(this);
    addDockWidget(Qt::TopDockWidgetArea, cake);
    initDocksWidget(cake, m_layoutRoot);
    m_nResizeTimes = 2;
}

void ZenoMainWindow::_resizeDocks(PtrLayoutNode root)
{
    if (!root)
        return;

    if (root->type == NT_ELEM)
    {
        if (root->geom.width() > 0) {
            int W = size().width() * root->geom.width();
            resizeDocks({root->pWidget}, {W}, Qt::Horizontal);
        }
        if (root->geom.height() > 0){
            int H = size().height() * root->geom.height();
            resizeDocks({root->pWidget}, {H}, Qt::Vertical);
        }
    }
    else
    {
        _resizeDocks(root->pLeft);
        _resizeDocks(root->pRight);
    }
}

void ZenoMainWindow::initDocksWidget(ZDockWidget* pLeft, PtrLayoutNode root)
{
    if (!root)
        return;

    if (root->type == NT_HOR || root->type == NT_VERT)
    {
        //skip optix view when enable ZENO_OPTIX_PROC
        ZDockWidget* pRight = new ZDockWidget(this);
        Qt::Orientation ori = root->type == NT_HOR ? Qt::Horizontal : Qt::Vertical;
        splitDockWidget(pLeft, pRight, ori);
        initDocksWidget(pLeft, root->pLeft);
        initDocksWidget(pRight, root->pRight);
    }
    else if (root->type == NT_ELEM)
    {
        int dockContentViewIndex = 0;
        root->pWidget = pLeft;
        for (QString tab : root->tabs)
        {
            PANEL_TYPE type = ZDockWidget::title2Type(tab);
            if (type != PANEL_EMPTY)
            {
                if ((type == PANEL_GL_VIEW || type == PANEL_OPTIX_VIEW) && root->widgetInfos.size() != 0)
                {
                    if (dockContentViewIndex < root->widgetInfos.size())
                        pLeft->onAddTab(type, root->widgetInfos[dockContentViewIndex++]);
                }
                else {
                    pLeft->onAddTab(type);
                }
            }
        }
    }
}

void ZenoMainWindow::initCustomLayoutAction(const QStringList &list, bool isDefault) {
    QList<QAction *> actions;
    for (QString name : list) {
        if (name == g_latest_layout) {
            continue;
        }
        QAction *pCustomLayout_ = new QAction(name);
        connect(pCustomLayout_, &QAction::triggered, this, [=]() { 
            loadDockLayout(name, isDefault); 
            updateLatestLayout(name);
        });
        actions.append(pCustomLayout_);
    }
    if (isDefault) {
        m_ui->menuWindow->insertActions(m_ui->actionSave_Layout, actions);
        m_ui->menuWindow->insertSeparator(m_ui->actionSave_Layout);
    } else {
        m_ui->menuCustom_Layout->addActions(actions);
    }
}

void ZenoMainWindow::loadDockLayout(QString name, bool isDefault) 
{
    QString content;
    if (isDefault) 
    {
        QJsonObject obj = readDefaultLayout();
        bool isSuccess = false;
        if (!name.isEmpty()) 
        {
            for (QJsonObject::const_iterator it = obj.constBegin(); it != obj.constEnd(); it++) 
            {
                if (it.key() == name) 
                {
                    QJsonObject layout = it.value().toObject();
                    QJsonDocument doc(layout);
                    content = doc.toJson();
                    isSuccess = true;
                    break;
                }
            }
        } 
        if (!isSuccess) 
        {
            QJsonObject layout = obj.constBegin().value().toObject();
            QJsonDocument doc(layout);
            content = doc.toJson();
        }
    } 
    else 
    {
        QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
        settings.beginGroup("layout");
        settings.beginGroup(name);
        if (settings.allKeys().indexOf("content") != -1) 
        {
            content = settings.value("content").toString();
            settings.endGroup();
            settings.endGroup();
        } 
        else
        {
            loadDockLayout(name, true);
            return;
        }
    }
    if (!content.isEmpty()) 
    {
        PtrLayoutNode root = readLayout(content);
        resetDocks(root);
    } 
    else 
    {
        QMessageBox msg(QMessageBox::Warning, "", tr("layout format is invalid."));
        msg.exec();
    }
}

QJsonObject ZenoMainWindow::readDefaultLayout() 
{
    QString filename = ":/templates/DefaultLayout.txt";
    QFile file(filename);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        return QJsonObject();
    }
    QByteArray byteArray = file.readAll();
    QJsonDocument doc = QJsonDocument::fromJson(byteArray);
    if (doc.isObject()) {
        return doc.object();
    }
    return QJsonObject();
}

void ZenoMainWindow::manageCustomLayout() 
{
    ZDockLayoutMangeDlg dlg(this);
    connect(&dlg, &ZDockLayoutMangeDlg::layoutChangedSignal, this, &ZenoMainWindow::loadSavedLayout);
    dlg.exec();
}

void ZenoMainWindow::updateLatestLayout(const QString &layout) 
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    settings.beginGroup(g_latest_layout);
    settings.setValue(g_latest_layout, layout);
    settings.endGroup();
    settings.endGroup();
}

void ZenoMainWindow::initDocks(PANEL_TYPE onlyView)
{
    if (onlyView != PANEL_EMPTY)
    {
        m_layoutRoot = std::make_shared<LayerOutNode>();
        m_layoutRoot->type = NT_ELEM;
        m_bOnlyOptix = onlyView == PANEL_OPTIX_VIEW;

        ZDockWidget* onlyWid = new ZDockWidget(this);
        if (onlyView == PANEL_GL_VIEW || onlyView == PANEL_OPTIX_VIEW)
            onlyWid->setCurrentWidget(onlyView);

        addDockWidget(Qt::TopDockWidgetArea, onlyWid);
        m_layoutRoot->type = NT_ELEM;
        m_layoutRoot->pWidget = onlyWid;

        initTimelineDock();

        return;
    }
    /*m_layoutRoot = std::make_shared<LayerOutNode>();
    m_layoutRoot->type = NT_ELEM;

    ZDockWidget* viewDock = new ZDockWidget(this);
    viewDock->setCurrentWidget(PANEL_VIEW);
    viewDock->setObjectName("viewDock");

    ZDockWidget *logDock = new ZDockWidget(this);
    logDock->setCurrentWidget(PANEL_LOG);
    logDock->setObjectName("logDock");

    ZDockWidget *paramDock = new ZDockWidget(this);
    paramDock->setCurrentWidget(PANEL_NODE_PARAMS);
    paramDock->setObjectName("paramDock");

    ZDockWidget* editorDock = new ZDockWidget(this);
    editorDock->setCurrentWidget(PANEL_EDITOR);
    editorDock->setObjectName("editorDock");

    addDockWidget(Qt::TopDockWidgetArea, viewDock);
    m_layoutRoot->type = NT_ELEM;
    m_layoutRoot->pWidget = viewDock;

    SplitDockWidget(viewDock, editorDock, Qt::Vertical);
    SplitDockWidget(viewDock, logDock, Qt::Horizontal);
    SplitDockWidget(editorDock, paramDock, Qt::Horizontal);

    //paramDock->hide();
    logDock->hide();*/

    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    settings.beginGroup(g_latest_layout);
    QString name;
    if (settings.allKeys().indexOf(g_latest_layout) != -1) {
        name = settings.value(g_latest_layout).toString();
    } 
    settings.endGroup();
    settings.endGroup();
    loadDockLayout(name, false);

    initTimelineDock();
}

void ZenoMainWindow::initTimelineDock()
{
    m_pTimeline = new ZTimeline;
    setCentralWidget(m_pTimeline);

    connect(m_pTimeline, &ZTimeline::playForward, this, [=](bool bPlaying) {
        QVector<DisplayWidget*> views = viewports();
        for (DisplayWidget* view : views) {
            view->onPlayClicked(bPlaying);
        }
    });

    connect(m_pTimeline, &ZTimeline::sliderValueChanged, this, &ZenoMainWindow::onFrameSwitched);

    auto graphs = zenoApp->graphsManager();
    connect(graphs, &GraphsManager::modelDataChanged, this, [=]() {
        if (!m_bAlways && !m_bAlwaysLightCamera && !m_bAlwaysMaterial)
            return;
#if 0
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        mgr->setCacheOpt(ZCacheMgr::Opt_AlwaysOn);
        m_pTimeline->togglePlayButton(false);
        int nFrame = m_pTimeline->value();
        QVector<DisplayWidget *> views = viewports();
        std::function<void(bool, bool)> setOptixUpdateSeparately = [=](bool updateLightCameraOnly, bool updateMatlOnly) {
            QVector<DisplayWidget *> views = viewports();
            for (auto displayWid : views) {
                if (!displayWid->isGLViewport()) {
                    displayWid->setRenderSeparately(updateLightCameraOnly, updateMatlOnly);
                }
            }
        };
        for (DisplayWidget *view : views) {
            if (m_bAlways) {
                setOptixUpdateSeparately(false, false);
                LAUNCH_PARAM launchParam;
                launchParam.beginFrame = nFrame;
                launchParam.endFrame = nFrame;
                launchParam.projectFps = timeline()->fps();
                AppHelper::initLaunchCacheParam(launchParam);
                view->onRun(launchParam);
            }
            else if (m_bAlwaysLightCamera || m_bAlwaysMaterial) {
                setOptixUpdateSeparately(m_bAlwaysLightCamera, m_bAlwaysMaterial);
                LAUNCH_PARAM launchParam;
                launchParam.beginFrame = nFrame;
                launchParam.endFrame = nFrame;
                launchParam.applyLightAndCameraOnly = m_bAlwaysLightCamera;
                launchParam.applyMaterialOnly = m_bAlwaysMaterial;
                launchParam.projectFps = timeline()->fps();
                AppHelper::initLaunchCacheParam(launchParam);
                view->onRun(launchParam);
            }
        }
#endif
    });
}

ZTimeline* ZenoMainWindow::timeline() const
{
    return m_pTimeline;
}

void ZenoMainWindow::onFrameSwitched(int frameid)
{
    auto& sess = zeno::getSession();
    sess.switchToFrame(frameid);
}

void ZenoMainWindow::onCalcFinished(bool bSucceed, zeno::ObjPath nodeUuidPath, QString msg)
{
    if (!bSucceed) {
        ZenoGraphsEditor* pEditor = getAnyEditor();
        if (pEditor) {
            GraphsTreeModel* pTreeM = zenoApp->graphsManager()->currentModel();
            if (pTreeM) {
                QModelIndex nodeIdx = pTreeM->getIndexByUuidPath(nodeUuidPath);
                QStringList nodePath = nodeIdx.data(ROLE_OBJPATH).toStringList();
                QString nodeName = nodePath.back();
                nodePath.pop_back();
                pEditor->activateTab(nodePath, nodeName, true);
            }
        }
    }
}

void ZenoMainWindow::onMaximumTriggered()
{
    ZDockWidget* pDockWidget = qobject_cast<ZDockWidget*>(sender());
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* pDock : docks)
    {
        if (pDock != pDockWidget)
        {
            pDock->close();
        }
    }
}

DisplayWidget *ZenoMainWindow::getOptixWidget() const
{
    auto views = viewports();
    for (auto view : views)
    {
        if (!view->isGLViewport())
            return view;
    }
    return nullptr;
}

QVector<DisplayWidget*> ZenoMainWindow::viewports() const
{
    QVector<DisplayWidget*> views;
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* pDock : docks)
    {
        if (pDock->isVisible())
            views.append(pDock->viewports());
    }

    //top level floating windows.
    QWidgetList lst = QApplication::topLevelWidgets();
    for (auto wid : lst)
    {
        if (ZDockWidget* pFloatWin = qobject_cast<ZDockWidget*>(wid))
        {
            views.append(pFloatWin->viewports());
        }
    }
    return views;
}

DisplayWidget* ZenoMainWindow::getCurrentViewport() const
{
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    QVector<DisplayWidget*> vec;
    for (ZDockWidget* pDock : docks)
    {
        if (pDock->isVisible())
        {
            if (ZDockTabWidget* tabwidget = qobject_cast<ZDockTabWidget*>(pDock->widget()))
            {
                if (DockContent_View* wid = qobject_cast<DockContent_View*>(tabwidget->currentWidget()))
                {
                    DisplayWidget* pView = wid->getDisplayWid();
                    if (pView && pView->isCurrent())
                    {
                        return pView;
                    }
                }
            }
        }
    }
    return nullptr;
}

void ZenoMainWindow::toggleTimelinePlay(bool bOn)
{
    m_pTimeline->togglePlayButton(bOn);
}

void ZenoMainWindow::onRunTriggered(bool applyLightAndCameraOnly, bool applyMaterialOnly)
{
    QVector<DisplayWidget*> views = viewports();

    clearErrorMark();

    //TODO: the run procedure shoule be designed carefully.

#if 0
    for (auto view : views)
    {
        view->beforeRun();
    }

    ZASSERT_EXIT(m_pTimeline);
    QPair<int, int> fromTo = m_pTimeline->fromTo();
    int beginFrame = fromTo.first;
    int endFrame = fromTo.second;
    if (endFrame >= beginFrame && beginFrame >= 0)
    {
        auto pGraphsMgr = zenoApp->graphsManager();
        GraphsTreeModel* pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        LAUNCH_PARAM launchParam;
        launchParam.beginFrame = beginFrame;
        launchParam.endFrame = endFrame;
        launchParam.applyLightAndCameraOnly = applyLightAndCameraOnly;
        launchParam.applyMaterialOnly = applyMaterialOnly;
        QString path = pModel->filePath();
        path = path.left(path.lastIndexOf("/"));
        launchParam.zsgPath = path;
        auto main = zenoApp->getMainWindow();
        ZASSERT_EXIT(main);
        launchParam.projectFps = main->timelineInfo().timelinefps;
        AppHelper::initLaunchCacheParam(launchParam);
        launchProgram(pModel, launchParam);
    }

    for (auto view : views)
    {
        view->afterRun();
    }
    if (m_pTimeline)
    {
        m_pTimeline->updateCachedFrame();
    }
#endif
}

DisplayWidget* ZenoMainWindow::getOnlyViewport() const
{
    //find the only optix viewport
    QVector<DisplayWidget*> views;
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* pDock : docks)
    {
        views.append(pDock->viewports());
    }
    ZASSERT_EXIT(views.size() == 1, nullptr);

    DisplayWidget* pView = views[0];
    return pView;
}

bool ZenoMainWindow::resetProc()
{
    //should kill the runner proc.
    const bool bWorking = zeno::getSession().globalState->is_working();
    if (bWorking)
    {
        int flag = QMessageBox::question(this, tr("Kill Process"), tr("Background process is running, you need kill the process."), QMessageBox::Yes | QMessageBox::No);
        if (flag & QMessageBox::Yes)
        {
            //TODO: kill current calculation
            //killProgram();
        }
        else
        {
            return false;
        }
    }
    return true;
}

void ZenoMainWindow::optixClientSend(QString& info)
{
    optixClientSocket->write(info.toUtf8());
}

void ZenoMainWindow::optixClientStartRec()
{
    m_bOptixProcRecording = true;
}

void ZenoMainWindow::updateViewport(const QString& action)
{
    QVector<DisplayWidget*> views = viewports();
    for (DisplayWidget* view : views)
    {
        view->updateFrame(action);
    }
    if (m_pTimeline)
    {
        if (action == "finishFrame")
        {
            updateLightList();
            bool bPlayed = m_pTimeline->isPlayToggled();
            int endFrame = zeno::getSession().globalComm->maxPlayFrames() - 1;
            m_pTimeline->updateCachedFrame();
            if (!bPlayed)
            {
                int ui_frame = m_pTimeline->value();
                if (ui_frame == endFrame)
                {
                    for (DisplayWidget* view : views)
                    {
                        if (view->isGLViewport())
                        {
                            Zenovis* pZenovis = view->getZenoVis();
                            ZASSERT_EXIT(pZenovis);
                            pZenovis->setCurrentFrameId(ui_frame);
                        }
                        else
                        {
#ifndef ZENO_OPTIX_PROC
                            ZOptixViewport* pOptix = view->optixViewport();
                            ZASSERT_EXIT(pOptix);
                            emit pOptix->sig_switchTimeFrame(ui_frame);
#else
                            ZOptixProcViewport* pOptix = view->optixViewport();
                            ZASSERT_EXIT(pOptix);
                            if (Zenovis* vis = pOptix->getZenoVis())
                            {
                                vis->setCurrentFrameId(ui_frame);
                                vis->startPlay(false);
                            }
#endif
                        }
                        view->updateFrame();
                    }
                }
            }
        }
    }
}

ZenoGraphsEditor* ZenoMainWindow::getAnyEditor() const
{
    auto docks2 = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto dock : docks2)
    {
        if (!dock->isVisible())
            continue;
        ZenoGraphsEditor* pEditor = dock->getAnyEditor();
        if (pEditor)
            return pEditor;
    }
    return nullptr;
}

void ZenoMainWindow::onRunFinished()
{
    auto docks2 = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto dock : docks2)
    {
        dock->onRunFinished();
    }
}

void ZenoMainWindow::onCloseDock()
{
    ZDockWidget *pDockWidget = qobject_cast<ZDockWidget *>(sender());
    ZASSERT_EXIT(pDockWidget);
    pDockWidget->close();
    //pDockWidget->testCleanupGL();

    PtrLayoutNode spParent = findParent(m_layoutRoot, pDockWidget);
    if (spParent)
    {
        if (spParent->pLeft->pWidget == pDockWidget)
        {
            PtrLayoutNode right = spParent->pRight;
            spParent->pWidget = right->pWidget;
            spParent->pLeft = right->pLeft;
            spParent->pRight = right->pRight;
            spParent->type = right->type;
        }
        else if (spParent->pRight->pWidget == pDockWidget)
        {
            PtrLayoutNode left = spParent->pLeft;
            spParent->pWidget = left->pWidget;
            spParent->pLeft = left->pLeft;
            spParent->pRight = left->pRight;
            spParent->type = left->type;
        }
    }
    else
    {
        m_layoutRoot = nullptr;
    }
}

void ZenoMainWindow::SplitDockWidget(ZDockWidget* after, ZDockWidget* dockwidget, Qt::Orientation orientation)
{
    splitDockWidget(after, dockwidget, orientation);

    PtrLayoutNode spRoot = findNode(m_layoutRoot, after);
    ZASSERT_EXIT(spRoot);

    spRoot->type = (orientation == Qt::Vertical ? NT_VERT : NT_HOR);
    spRoot->pWidget = nullptr;

    spRoot->pLeft = std::make_shared<LayerOutNode>();
    spRoot->pLeft->pWidget = after;
    spRoot->pLeft->type = NT_ELEM;

    spRoot->pRight = std::make_shared<LayerOutNode>();
    spRoot->pRight->pWidget = dockwidget;
    spRoot->pRight->type = NT_ELEM;
}

void ZenoMainWindow::onSplitDock(bool bHorzontal)
{
    ZDockWidget* pDockWidget = qobject_cast<ZDockWidget*>(sender());
    ZDockWidget* pDock = new ZDockWidget(this);

    //QLayout* pLayout = this->layout();
    //QMainWindowLayout* pWinLayout = qobject_cast<QMainWindowLayout*>(pLayout);

    pDock->setCurrentWidget(PANEL_EDITOR);
    //pDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    SplitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
}

void ZenoMainWindow::openFileDialog()
{
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;
    if (!resetProc())
        return;
    //todo: path validation
    if (saveQuit()) 
    {
        openFile(filePath);
    }
}

void ZenoMainWindow::onNewFile() {
    if (!resetProc())
        return;
    if (saveQuit()) 
    {
        zenoApp->graphsManager()->newFile();
    }
}

void ZenoMainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
}

void ZenoMainWindow::killOptix()
{
    DisplayWidget* pWid = this->getOptixWidget();
    if (pWid)
    {
        pWid->killOptix();
    }
    if (optixClientSocket)
    {
        optixClientSocket->write(std::string("optixProcClose").data());
        if (!optixClientSocket->waitForBytesWritten(50000))
        {
            zeno::log_error("tcp optix client close fail");
        }
    }
}

void ZenoMainWindow::closeEvent(QCloseEvent *event)
{
    //killProgram();
    killOptix();

    QSettings settings(zsCompanyName, zsEditor);
    bool autoClean = settings.value("zencache-autoclean").isValid() ? settings.value("zencache-autoclean").toBool() : true;
    bool autoRemove = settings.value("zencache-autoremove").isValid() ? settings.value("zencache-autoremove").toBool() : false;

    //TODO: clean issues.

    bool isClose = this->saveQuit();
    // todo: event->ignore() when saveQuit returns false?
    if (isClose) 
    {
        //save latest layout
        QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
        settings.beginGroup("layout");
        QString layoutInfo = exportLayout(m_layoutRoot, size());
        settings.beginGroup(g_latest_layout);
        settings.setValue("content", layoutInfo);
        settings.endGroup();
        settings.endGroup();

        //clean up opengl components.

        auto docks = findChildren<ZDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
        for (ZDockWidget *pDock : docks) {
            pDock->close();
            try {
                //pDock->testCleanupGL();
            } catch (...) {
                //QString errMsg = QString::fromLatin1(e.what());
                int j;
                j = 0;
            }
            //delete pDock;
        }

        // trigger destroy event
        zeno::getSession().eventCallbacks->triggerEvent("beginDestroy");

        QMainWindow::closeEvent(event);
    } 
    else 
    {
        event->ignore();
    }
}

bool ZenoMainWindow::event(QEvent* event)
{
    if (QEvent::LayoutRequest == event->type())
    {
        //resizing have to be done after fitting layout, which follows by LayoutRequest.
        //it seems that after `m_nResizeTimes` times, the resize action can be valid...
        if (m_nResizeTimes > 0 && m_layoutRoot)
        {
            --m_nResizeTimes;
            if (m_nResizeTimes == 0)
            {
                _resizeDocks(m_layoutRoot);
                return true;
            }
        }
    }
    if (event->type() == QEvent::HoverMove) {
        if (m_bOnlyOptix) {
            DisplayWidget* pWid = getCurrentViewport();
            pWid->onMouseHoverMoved();
        }
    }
    return QMainWindow::event(event);
}

void ZenoMainWindow::mousePressEvent(QMouseEvent* event)
{
    QMainWindow::mousePressEvent(event);
}

void ZenoMainWindow::mouseMoveEvent(QMouseEvent* event)
{
    QMainWindow::mouseMoveEvent(event);
}

void ZenoMainWindow::mouseReleaseEvent(QMouseEvent* event)
{
    QMainWindow::mouseReleaseEvent(event);
}

void ZenoMainWindow::dragEnterEvent(QDragEnterEvent* event)
{
    auto urls = event->mimeData()->urls();
    if (urls.size() == 1 && urls[0].toLocalFile().endsWith(".zsg")) {
        event->acceptProposedAction();
    }
}

void ZenoMainWindow::dropEvent(QDropEvent* event)
{
    auto urls = event->mimeData()->urls();
    if (urls.size() != 1) {
        return;
    }
    auto filePath = urls[0].toLocalFile();
    if (!filePath.endsWith(".zsg")) {
        return;
    }

    if (saveQuit()) {
        openFile(filePath);
    }
}

void ZenoMainWindow::onZenovisFrameUpdate(bool bGLView, int frameid)
{
    if (!m_pTimeline)
        return;
    m_pTimeline->onTimelineUpdate(frameid);
}

void ZenoMainWindow::onCheckUpdate()
{
#ifdef __linux__
    return;
#else
    ZCheckUpdateDlg dlg(this);
    connect(&dlg, &ZCheckUpdateDlg::updateSignal, this, [=](const QString& version, const QString &url) {
        auto pGraphsMgm = zenoApp->graphsManager();
        ZASSERT_EXIT(pGraphsMgm, true);
        GraphsTreeModel* pModel = pGraphsMgm->currentModel();
        bool bUpdate = true;
        if (!zeno::envconfig::get("OPEN") && pModel && pModel->isDirty()) 
        {
            ZRestartDlg restartDlg;
            connect(&restartDlg, &ZRestartDlg::saveSignal, [=](bool bSaveAs) {
                if (bSaveAs)
                {
                    saveAs();
                }
                else
                {
                    save();
                }
            });
            if (restartDlg.exec() != QDialog::Accepted)
            {
                bUpdate = false;
            }
        }
        if (bUpdate)
        {
            //start install proc
            QString sCmd = "--version " + version + " --url " + url;
            ShellExecuteA(NULL, NULL, "zenoinstall.exe", sCmd.toLocal8Bit().data(), NULL, SW_SHOWNORMAL);

            //killProgram();
            //killOptix();
            zeno::getSession().eventCallbacks->triggerEvent("beginDestroy");
            zenoApp->quit();
        }
    });
    connect(&dlg, &ZCheckUpdateDlg::remindSignal, this, [=]() {
        QTimer timer;
        //10mins remind
        timer.singleShot(10 * 60 * 1000, this, &ZenoMainWindow::onCheckUpdate);
    });
    dlg.exec();
#endif
}

void ZenoMainWindow::importGraph(bool bPreset)
{
    //in the new arch, import Graph means import assets.
    //TODO:
#if 0
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;

    //todo: path validation
    auto pGraphs = zenoApp->graphsManager();
    QMap<QString, QString> subgraphNames;//old name: new name
    QFile file(filePath);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (ret) {
        rapidjson::Document doc;
        QByteArray bytes = file.readAll();
        doc.Parse(bytes);

        if (doc.IsObject()  && doc.HasMember("graph"))
        {
            const rapidjson::Value& graph = doc["graph"];
            if (!graph.IsNull()) {
                GraphsTreeModel *pModel = pGraphs->currentModel();
                if (pModel)
                {
                    QStringList subgraphLst = pModel->subgraphsName();
                    QStringList duplicateLst;
                    for (const auto& subgraph : graph.GetObject())
                    {
                        const QString& graphName = subgraph.name.GetString();
                        if (graphName == "main")
                            continue;
                        if (subgraphLst.contains(graphName))
                        {
                            duplicateLst << graphName;
                        }
                        else
                        {
                            subgraphNames[graphName] = graphName;
                        }
                    }
                    if (!duplicateLst.isEmpty())
                    {
                        ZImportSubgraphsDlg dlg(duplicateLst, this);
                        connect(&dlg, &ZImportSubgraphsDlg::selectedSignal, this, [&subgraphNames, subgraphLst](const QStringList& lst, bool bRename) mutable {
                            if (!lst.isEmpty())
                            {
                                for (const QString name : lst)
                                {
                                    if (bRename)
                                    {
                                        QString newName = name;
                                        int i = 1;
                                        while (subgraphLst.contains(newName))
                                        {
                                            newName = name + QString("_%1").arg(i);
                                            i++;
                                        }
                                        subgraphNames[name] = newName;
                                    }
                                    else
                                        subgraphNames[name] = name;
                                }
                            }
                        });
                        dlg.exec();
                    }
                }
            }
        }
    }
    if (!subgraphNames.isEmpty())
        pGraphs->importSubGraphs(filePath, subgraphNames);
    if (bPreset)
    {
        for (const auto& name : subgraphNames)
        {
            QModelIndex index = pGraphs->currentModel()->index(name);
            if (index.isValid())
            {
                pGraphs->currentModel()->setData(index, SUBGRAPH_PRESET, ROLE_SUBGRAPH_TYPE);
            }
        }
        ZenoSettingsManager::GetInstance().setValue(zsSubgraphType, SUBGRAPH_PRESET);
    }
    else
    {
        ZenoSettingsManager::GetInstance().setValue(zsSubgraphType, SUBGRAPH_NOR);
    }
#endif
}

static bool saveContent(const QString &strContent, QString filePath) {
    QFile f(filePath);
    zeno::log_debug("saving {} chars to file [{}]", strContent.size(), filePath.toStdString());
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << f.errorString();
        zeno::log_error("Failed to open file for write: {} ({})", filePath.toStdString(),
                        f.errorString().toStdString());
        return false;
    }
    f.write(strContent.toUtf8());
    f.close();
    zeno::log_debug("saved successfully");
    return true;
}

void ZenoMainWindow::exportGraph()
{
    //TODO: export the assets?
#if 0
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Export", "",
                                                "C++ Source File(*.cpp);; JSON file(*.json);; All Files(*);;");
    if (path.isEmpty()) {
        return;
    }

    //auto pGraphs = zenoApp->graphsManager();
    //pGraphs->importGraph(path);

    QString content;
    {
        GraphsTreeModel *pModel = zenoApp->graphsManager()->currentModel();
        if (path.endsWith(".cpp")) {
            content = serializeSceneCpp(pModel);
        } else {
            rapidjson::StringBuffer s;
            RAPIDJSON_WRITER writer(s);
            writer.StartArray();
            LAUNCH_PARAM launchParam;
            serializeScene(pModel, writer, launchParam);
            writer.EndArray();
            content = QString(s.GetString());
        }
    }
    saveContent(content, path);
#endif
}

bool ZenoMainWindow::openFile(QString filePath)
{
    auto pGraphs = zenoApp->graphsManager();
    zenoio::ERR_CODE code = zenoio::PARSE_NOERROR;
    GraphsTreeModel* pModel = pGraphs->openZsgFile(filePath, code);
    if (!pModel)
    {
        if (code == zenoio::PARSE_VERSION_UNKNOWN) {
            QMessageBox::information(this, tr("Open File"), tr("The format of file is unknown"));
        }
        return false;
    }

    //cleanup
    zeno::getSession().globalComm->clearFrameState();
    auto views = viewports();
    for (auto view : views)
    {
        view->cleanUpScene();
    }

    resetTimeline(pGraphs->timeInfo());
    recordRecentFile(filePath);
    initUserdata(pGraphs->userdataInfo());
    //resetDocks(pGraphs->layoutInfo().layerOutNode);

    m_ui->statusbar->showMessage(tr("File Opened"));
    zeno::scope_exit sp([&]() {QTimer::singleShot(2000, this, [=]() {m_ui->statusbar->showMessage(tr("Status Bar")); }); });
    return true;
}

void ZenoMainWindow::loadRecentFiles() 
{
    m_ui->menuRecent_Files->clear();
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");
    QStringList lst = settings.childKeys();
    sortRecentFile(lst);
    for (int i = 0; i < lst.size(); i++) {
        const QString &key = lst[i];
        const QString &path = settings.value(key).toString();
        if (!path.isEmpty()) {
            QAction *action = new QAction(path);
            m_ui->menuRecent_Files->addAction(action);
            connect(action, &QAction::triggered, this, [=]() {
                if (!resetProc())
                    return;
                if (saveQuit()) {
                    if (!QFileInfo::exists(path)) {
                        int flag = QMessageBox::question(nullptr, "", tr("the file does not exies, do you want to remove it?"), QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                        if (flag & QMessageBox::Yes) {
                            QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                            _settings.beginGroup("Recent File List");
                            _settings.remove(key);
                            m_ui->menuRecent_Files->removeAction(action);
                            emit recentFilesChanged(this);
                        }
                    } else {
                        openFile(path);
                    }
                }
            });
        }
    }
}

void ZenoMainWindow::initShortCut() 
{
    QStringList lst;
    lst << ShortCut_Open /*<< ShortCut_Save << ShortCut_SaveAs*/ << ShortCut_Import 
        << ShortCut_Export_Graph << ShortCut_Undo << ShortCut_Redo << ShortCut_ScreenShoot << ShortCut_RecordVideo
        << ShortCut_NewSubgraph << ShortCut_New_File;
    updateShortCut(lst);
    connect(&ZenoSettingsManager::GetInstance(), &ZenoSettingsManager::valueChanged, this, [=](QString key) {
        updateShortCut(QStringList(key));
    });
}

void ZenoMainWindow::updateShortCut(QStringList keys)
{
    ZenoSettingsManager &settings = ZenoSettingsManager::GetInstance();
    if (keys.contains(ShortCut_Open))
        m_ui->action_Open->setShortcut(settings.getShortCut(ShortCut_Open));
    if (keys.contains(ShortCut_New_File))
        m_ui->actionNew_File->setShortcut(settings.getShortCut(ShortCut_New_File));
    if (keys.contains(ShortCut_Save))
        m_ui->action_Save->setShortcut(settings.getShortCut(ShortCut_Save));
    if (keys.contains(ShortCut_SaveAs))
        m_ui->action_Save_As->setShortcut(settings.getShortCut(ShortCut_SaveAs));
    if (keys.contains(ShortCut_Import))
        m_ui->action_Import->setShortcut(settings.getShortCut(ShortCut_Import));
    if (keys.contains(ShortCut_Export_Graph))
        m_ui->actionExportGraph->setShortcut(settings.getShortCut(ShortCut_Export_Graph));
     if (keys.contains(ShortCut_Undo))
        m_ui->actionUndo->setShortcut(settings.getShortCut(ShortCut_Undo));
    if (keys.contains(ShortCut_Redo))
        m_ui->actionRedo->setShortcut(settings.getShortCut(ShortCut_Redo));
    if (keys.contains(ShortCut_ScreenShoot))
        m_ui->actionScreen_Shoot->setShortcut(settings.getShortCut(ShortCut_ScreenShoot));
    if (keys.contains(ShortCut_RecordVideo))
        m_ui->actionRecord_Video->setShortcut(settings.getShortCut(ShortCut_RecordVideo));
    if (keys.contains(ShortCut_NewSubgraph))
        m_ui->actionNew_Asset->setShortcut(settings.getShortCut(ShortCut_NewSubgraph));
}

void ZenoMainWindow::shortCutDlg() 
{
    ZShortCutSettingDlg dlg;
    dlg.exec();
}

bool ZenoMainWindow::isOnlyOptixWindow() const
{
    return m_bOnlyOptix;
}

bool ZenoMainWindow::isRecordByCommandLine() const
{
    return m_bRecordByCommandLine;
}

void ZenoMainWindow::sortRecentFile(QStringList &lst) 
{
    qSort(lst.begin(), lst.end(), [](const QString &s1, const QString &s2) {
        static QRegExp rx("File (\\d+)");
        int num1 = 0;
        if (rx.indexIn(s1) != -1) {
            QStringList caps = rx.capturedTexts();
            if (caps.length() == 2)
                num1 = caps[1].toInt();
        }
        int num2 = 0;
        if (rx.indexIn(s2) != -1) {
            QStringList caps = rx.capturedTexts();
            if (caps.length() == 2)
                num2 = caps[1].toInt();
        }
        return num1 > num2;
    });
}

void ZenoMainWindow::recordRecentFile(const QString& filePath)
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");

    QStringList keys = settings.childKeys();
    sortRecentFile(keys);
    QStringList paths;
    for (QString key : keys) {
        QString path = settings.value(key).toString();
        if (path == filePath)
        {
            //remove the old record.
            settings.remove(key);
            continue;
        }
        paths.append(path);
    }

    if (paths.indexOf(filePath) != -1) {
        return;
    }

    int idx = -1;
    if (keys.isEmpty()) {
        idx = 0;
    } else {
        QString key = keys.first();
        static QRegExp rx("File (\\d+)");
        if (rx.indexIn(key) != -1) 
        {
            QStringList caps = rx.capturedTexts();
            if (caps.length() == 2 && idx < caps[1].toInt())
                idx = caps[1].toInt();
        }
    }

    settings.setValue(QString("File %1").arg(idx + 1), filePath);
    //limit 5
    while (settings.childKeys().size() > 5) {
        settings.remove(keys.last());
        keys.removeLast();
    }
    loadRecentFiles();
    emit recentFilesChanged(this);
}

void ZenoMainWindow::setActionProperty() 
{
    m_ui->actionNew_File->setProperty("ActionType", ACTION_NEWFILE);
    m_ui->actionNew_Asset->setProperty("ActionType", ACTION_NEW_SUBGRAPH);
    m_ui->action_Open->setProperty("ActionType", ACTION_OPEN);
    m_ui->action_Save->setProperty("ActionType", ACTION_SAVE);
    m_ui->action_Save_As->setProperty("ActionType", ACTION_SAVE_AS);
    m_ui->action_Import->setProperty("ActionType", ACTION_IMPORT);
    m_ui->actionExportGraph->setProperty("ActionType", ACTION_EXPORT_GRAPH);
    m_ui->actionScreen_Shoot->setProperty("ActionType", ACTION_SCREEN_SHOOT);
    m_ui->actionRecord_Video->setProperty("ActionType", ACTION_RECORD_VIDEO);
    m_ui->action_Close->setProperty("ActionType", ACTION_CLOSE);
    m_ui->actionUndo->setProperty("ActionType", ACTION_UNDO);
    m_ui->actionRedo->setProperty("ActionType", ACTION_REDO);
    m_ui->action_Copy->setProperty("ActionType", ACTION_COPY);
    m_ui->action_Paste->setProperty("ActionType", ACTION_PASTE);
    m_ui->action_Cut->setProperty("ActionType", ACTION_CUT);
    m_ui->actionCollaspe->setProperty("ActionType", ACTION_COLLASPE);
    m_ui->actionExpand->setProperty("ActionType", ACTION_EXPAND);
    m_ui->actionEasy_Graph->setProperty("ActionType", ACTION_EASY_GRAPH);
    m_ui->actionOpen_View->setProperty("ActionType", ACTION_OPEN_VIEW);
    m_ui->actionClear_View->setProperty("ActionType", ACTION_CLEAR_VIEW);
    m_ui->actionBlackWhite->setProperty("ActionType", ACTION_BLACK_WHITE);
    m_ui->actionCreek->setProperty("ActionType", ACTION_GREEK);
    m_ui->actionDay_Light->setProperty("ActionType", ACTION_DAY_LIGHT);
    m_ui->actionDefault->setProperty("ActionType", ACTION_DEFAULT);
    m_ui->actionFootballField->setProperty("ActionType", ACTION_FOOTBALL_FIELD);
    m_ui->actionForest->setProperty("ActionType", ACTION_FOREST);
    m_ui->actionLake->setProperty("ActionType", ACTION_LAKE);
    m_ui->actionSee->setProperty("ActionType", ACTION_SEA);
    m_ui->actionNode_Camera->setProperty("ActionType", ACTION_NODE_CAMERA);
    m_ui->actionSave_Layout->setProperty("ActionType", ACTION_SAVE_LAYOUT);
    m_ui->actionLayout_Manager->setProperty("ActionType", ACTION_LAYOUT_MANAGE);
    m_ui->actionEnglish_Chinese->setProperty("ActionType", ACTION_LANGUAGE);
    m_ui->actionShortcutList->setProperty("ActionType", ACTION_SHORTCUTLIST);
    m_ui->actionSet_NASLOC->setProperty("ActionType", ACTION_SET_NASLOC);
    m_ui->actionSet_ZENCACHE->setProperty("ActionType", ACTION_ZENCACHE);
    m_ui->actionSet_ShortCut->setProperty("ActionType", ACTION_SET_SHORTCUT);
    m_ui->actionPreferences->setProperty("ActionType", ACTION_PREFERENCES);
    m_ui->actionFeedback->setProperty("ActionType", ACTION_FEEDBACK);
    m_ui->actionAbout->setProperty("ActionType", ACTION_ABOUT);
    m_ui->actionCheck_Update->setProperty("ActionType", ACTION_CHECKUPDATE);
}

void ZenoMainWindow::screenShoot() 
{
    QString path = QFileDialog::getSaveFileName(
        nullptr, tr("Path to Save"), "",
        tr("PNG images(*.png);;JPEG images(*.jpg);;BMP images(*.bmp);;EXR images(*.exr);;HDR images(*.hdr);;"));
    QString ext = QFileInfo(path).suffix();
    if (!path.isEmpty()) {

        ZenoMainWindow *pWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(pWin);
        QVector<DisplayWidget*> views = pWin->viewports();
        if (!views.isEmpty())
        {
            //todo: ask the user to select a viewport to screenshot.
            DisplayWidget* pWid = views[0];
            ZASSERT_EXIT(pWid);
            auto pZenovis = pWid->getZenoVis();
            ZASSERT_EXIT(pZenovis);
            pZenovis->getSession()->do_screenshot(path.toStdString(), ext.toStdString());
        }
    }
}

#if 0
//todo: resolve conflict.
QString ZenoMainWindow::uniqueDockObjName(DOCK_TYPE type)
{
    switch (type)
    {
    case DOCK_EDITOR: return UiHelper::generateUuid("dock_editor_");
    case DOCK_LOG: return UiHelper::generateUuid("dock_log_");
    case DOCK_NODE_DATA: return UiHelper::generateUuid("dock_data_");
    case DOCK_VIEW: return UiHelper::generateUuid("dock_view_");
    case DOCK_NODE_PARAMS: return UiHelper::generateUuid("dock_parameter_");
    case DOCK_LIGHTS: return UiHelper::generateUuid("dock_lights_");
    default:
        return UiHelper::generateUuid("dock_empty_");
}
#endif

void ZenoMainWindow::setActionIcon(QAction *action) 
{
    if (!action->isCheckable() || !action->isChecked()) 
    {
        action->setIcon(QIcon());
#if 0
        case DOCK_IMAGE: {
            ZenoImagePanel* pPanel = new ZenoImagePanel;
            pDock->setWidget(type, pPanel);
            break;
        }
#endif
    }
    if (action->isChecked()) 
    {
        action->setIcon(QIcon("://icons/checked.png"));
    }
}

bool ZenoMainWindow::saveQuit() {
    auto pGraphsMgm = zenoApp->graphsManager();
    ZASSERT_EXIT(pGraphsMgm, true);
    GraphsTreeModel* pModel = pGraphsMgm->currentModel();
    if (!zeno::envconfig::get("OPEN") /* <- don't annoy me when I'm debugging via ZENO_OPEN */ && pModel && pModel->isDirty()) {
        QMessageBox msgBox(QMessageBox::Question, tr("Save"), tr("Save changes?"), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, this);
        QPalette pal = msgBox.palette();
        pal.setBrush(QPalette::WindowText, QColor(0, 0, 0));
        msgBox.setPalette(pal);
        int ret = msgBox.exec();
        if (ret & QMessageBox::Yes) {
            save();
        }
        if (ret & QMessageBox::Cancel) {
            return false;
        }
    }
    pGraphsMgm->clear();
    //clear timeline info.
    resetTimeline(zeno::TimelineInfo());
    return true;
}

void ZenoMainWindow::saveQuitShowWelcom()
{
    saveQuit();
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* pDock : docks)
        if (pDock->isVisible())
            if (ZDockTabWidget* tabwidget = qobject_cast<ZDockTabWidget*>(pDock->widget()))
                for (int i = 0; i < tabwidget->count(); i++)
                    if (DockContent_Editor* pEditor = qobject_cast<DockContent_Editor*>(tabwidget->widget(i)))
                        if (ZenoGraphsEditor* editor = pEditor->getEditor())
                            editor->showWelcomPage();

}

void ZenoMainWindow::save()
{
    auto pGraphsMgm = zenoApp->graphsManager();
    ZASSERT_EXIT(pGraphsMgm);
    GraphsTreeModel* pModel = pGraphsMgm->currentModel();
    if (!pModel)
        return;

    /*
    if (pModel->hasNotDescNode())
    {
        int flag = QMessageBox::question(this, "",
            tr("there is some nodes which are not descriped by the current version\n"
                "the save action will lose them, we recommand you choose \"Save As\" to save it"),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (flag & QMessageBox::No)
        {
            return;
        }
        saveAs();
        return;
    }
    */

    zeno::ZSG_VERSION ver = pGraphsMgm->ioVersion();
    if (zeno::VER_3 != ver)
    {
        QMessageBox msgBox(QMessageBox::Information, "", tr("The format of current zsg is old. To keep this file data trackable, we recommand you choose \"Save As\" to save it, as the format of new zsg"));
        msgBox.exec();
        bool ret = saveAs();
        if (ret) {
            pGraphsMgm->setIOVersion(zeno::VER_3);
        }
    }
    else
    {
        if (pModel)
        {
            QString currFilePath = pGraphsMgm->zsgPath();
            if (currFilePath.isEmpty())
                saveAs();
            else
                saveFile(currFilePath);
        }
    }
}

bool ZenoMainWindow::saveFile(QString filePath)
{
    const auto graphsMgr = zenoApp->graphsManager();
    APP_SETTINGS settings;
    settings.timeline = timelineInfo();
    settings.recordInfo = graphsMgr->recordSettings();
    settings.layoutInfo.layerOutNode = m_layoutRoot;
    settings.layoutInfo.size = size();
    settings.layoutInfo.cbDumpTabsToZsg = &AppHelper::dumpTabsToZsg;
    auto& ud = zeno::getSession().userData();
    settings.userdataInfo.optix_show_background = ud.get2<bool>("optix_show_background", false);
    graphsMgr->saveFile(filePath, settings);
    recordRecentFile(filePath);

    m_ui->statusbar->showMessage(tr("File Saved"));
    zeno::scope_exit sp([&]() {QTimer::singleShot(2000, this, [=]() {m_ui->statusbar->showMessage(tr("Status Bar")); }); });
    return true;
}

bool ZenoMainWindow::inDlgEventLoop() const {
    return m_bInDlgEventloop;
}

void ZenoMainWindow::setInDlgEventLoop(bool bOn) {
    m_bInDlgEventloop = bOn;
}

zeno::TimelineInfo ZenoMainWindow::timelineInfo()
{
    zeno::TimelineInfo info;
    ZASSERT_EXIT(m_pTimeline, info);
    info.bAlways = m_bAlways;
    info.beginFrame = m_pTimeline->fromTo().first;
    info.endFrame = m_pTimeline->fromTo().second;
    info.timelinefps = m_pTimeline->fps();
    info.currFrame = m_pTimeline->value();
    return info;
}

bool ZenoMainWindow::isAlways() const
{
    return m_bAlways;
}

bool ZenoMainWindow::isAlwaysLightCamera() const {
    return m_bAlwaysLightCamera;
}

bool ZenoMainWindow::isAlwaysMaterial() const {
    return m_bAlwaysMaterial;
}

void ZenoMainWindow::setAlways(bool bAlways)
{
    m_bAlways = bAlways;
    emit alwaysModeChanged(bAlways);
    if (m_bAlways)
        m_pTimeline->togglePlayButton(false);
}

void ZenoMainWindow::setAlwaysLightCameraMaterial(bool bAlwaysLightCamera, bool bAlwaysMaterial) {
    m_bAlwaysLightCamera = bAlwaysLightCamera;
    m_bAlwaysMaterial = bAlwaysMaterial;
}

void ZenoMainWindow::resetTimeline(zeno::TimelineInfo info)
{
    info.timelinefps = info.timelinefps < 1 ? 1 : info.timelinefps;
    setAlways(info.bAlways);
    m_pTimeline->initFromTo(info.beginFrame, info.endFrame);
    m_pTimeline->initFps(info.timelinefps);
    for (auto view: viewports())
    {
        view->setSliderFeq(1000 / info.timelinefps);
    }
}

void ZenoMainWindow::initUserdata(USERDATA_SETTING info)
{
    auto& ud = zeno::getSession().userData();
    ud.set2("optix_show_background", info.optix_show_background);
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    QVector<DisplayWidget*> vec;
    for (ZDockWidget* pDock : docks)
    {
        if (pDock->isVisible())
        {
            if (ZDockTabWidget* tabwidget = qobject_cast<ZDockTabWidget*>(pDock->widget()))
            {
                for (int i = 0; i < tabwidget->count(); i++)
                {
                    QWidget* wid = tabwidget->widget(i);
                    if (DockContent_View* pView = qobject_cast<DockContent_View*>(wid))
                        pView->setOptixBackgroundState(info.optix_show_background);
                }
            }
        }
    }
}

void ZenoMainWindow::onFeedBack()
{
    ZFeedBackDlg dlg(this);
    if (dlg.exec() == QDialog::Accepted)
    {
        QString content = dlg.content();
        bool isSend = dlg.isSendFile();
        if (isSend)
        {
            GraphsTreeModel *pModel = zenoApp->graphsManager()->currentModel();
            if (!pModel) {
                return;
            }
            APP_SETTINGS settings;
            //QString strContent = ZsgWriter::getInstance().dumpProgramStr(pModel, settings);
            //dlg.sendEmail("bug feedback", content, strContent);
        }
    }
}

void ZenoMainWindow::clearErrorMark()
{
    //clear all error mark at every scene.
    auto graphsMgm = zenoApp->graphsManager();
    graphsMgm->clearMarkOnGv();
}

bool ZenoMainWindow::saveAs() {
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zeno File(*.zen);; All Files(*);;");
    if (!path.isEmpty()) {
        return saveFile(path);
    }
    return false;
}

QString ZenoMainWindow::getOpenFileByDialog() {
    DlgInEventLoopScope;
    const QString &initialPath = "";
    QFileDialog fileDialog(this, tr("Open"), initialPath, "Legacy Zeno Graph File (*.zsg)\nZeno File (*.zen)\nAll Files (*)");
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setFileMode(QFileDialog::ExistingFile);
    if (fileDialog.exec() != QDialog::Accepted)
        return "";

    QString filePath = fileDialog.selectedFiles().first();
    return filePath;
}

void ZenoMainWindow::onNodesSelected(GraphModel* subgraph, const QModelIndexList &nodes, bool select) {
    //dispatch to all property panel.
    auto docks = findChildren<ZDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->onNodesSelected(subgraph, nodes, select);
    }
}

void ZenoMainWindow::onPrimitiveSelected(const std::unordered_set<std::string>& primids) {
    //dispatch to all property panel.
    auto docks = findChildren<ZDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->onPrimitiveSelected(primids);
    }
}

void ZenoMainWindow::updateLightList() {
    auto docks = findChildren<ZDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->updateLights();
    }
}
void ZenoMainWindow::doFrameUpdate(int frame) {
    //TODO: deprecated.
    std::cout << "====== Frame " << frame << "\n";

    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    for (auto displayWid : views)
    {
        ZASSERT_EXIT(displayWid);
        bool bMovingCamera = displayWid->isCameraMoving();
        std::cout << "====== CameraMoving " << bMovingCamera << "\n";

        // Sync Camera
        if (bMovingCamera) {

        }
        // Sync Frame
        else {
        }
    }
}

static bool openFileAndExportAsZsl(const char *inPath, const char *outPath) {
    //TODO: deprecated.
#if 0
    auto pGraphs = zenoApp->graphsManager();
    GraphsTreeModel* pModel = pGraphs->openZsgFile(inPath);
    if (!pModel) {
        qWarning() << "cannot open zsg file" << inPath;
        return false;
    }
    {
        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        writer.StartArray();
        LAUNCH_PARAM launchParam;
        serializeScene(pModel, writer, launchParam);
        writer.EndArray();
        QFile fout(outPath);
        /* printf("sadfkhjl jghkasdf [%s]\n", s.GetString()); */
        if (!fout.open(QIODevice::WriteOnly)) {
            qWarning() << "failed to open out zsl" << outPath;
            return false;
        }
        fout.write(s.GetString(), s.GetLength());
        fout.close();
    }
#endif
    return true;
}

static int subprogram_dumpzsg2zsl_main(int argc, char **argv) {
    //TODO: deprecated.
#if 0
    if (!argv[1]) {
        qWarning() << "please specify input zsg file path";
        return -1;
    }
    if (!argv[2]) {
        qWarning() << "please specify output zsl file path";
        return -1;
    }
    if (!openFileAndExportAsZsl(argv[1], argv[2])) {
        qWarning() << "failed to convert zsg to zsl";
        return -1;
    }
#endif
    return 0;
}

static int defDumpZsgToZslInit = zeno::getSession().eventCallbacks->hookEvent("init", [] (auto _) {
    zeno::getSession().userData().set("subprogram_dumpzsg2zsl", std::make_shared<zeno::GenericObject<int(*)(int, char **)>>(subprogram_dumpzsg2zsl_main));
});
