#include "launch/livehttpserver.h"
#include "launch/livetcpserver.h"
#include "zenomainwindow.h"
#include "dock/zenodockwidget.h"
#include <zenomodel/include/graphsmanagment.h>
#include "launch/corelaunch.h"
#include "launch/serialize.h"
#include "nodesview/zenographseditor.h"
#include "dock/ztabdockwidget.h"
#include "dock/docktabcontent.h"
#include "panel/zenodatapanel.h"
#include "panel/zenoproppanel.h"
#include "panel/zenospreadsheet.h"
#include "panel/zlogpanel.h"
#include "timeline/ztimeline.h"
#include "tmpwidgets/ztoolbar.h"
#include "viewport/viewportwidget.h"
#include "viewport/zenovis.h"
#include "zenoapplication.h"
#include <zeno/utils/log.h>
#include <zeno/utils/envconfig.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/writer/zsgwriter.h>
#include <zeno/core/Session.h>
#include <zenovis/DrawOptions.h>
#include <zenomodel/include/modeldata.h>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/uihelper.h>
#include "util/log.h"
#include "dialog/zfeedbackdlg.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"
#include "panel/zenolights.h"
#include "nodesys/zenosubgraphscene.h"
#include "viewport/recordvideomgr.h"
#include "ui_zenomainwindow.h"
#include <QJsonDocument>
#include "dialog/zdocklayoutmangedlg.h"


const QString g_latest_layout = "LatestLayout";

ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , m_bInDlgEventloop(false)
    , m_bAlways(false)
    , m_pTimeline(nullptr)
    , m_layoutRoot(nullptr)
    , m_nResizeTimes(0)
    , m_spCacheMgr(nullptr)
{
    liveTcpServer = new LiveTcpServer;
    liveHttpServer = new LiveHttpServer;
    liveSignalsBridge = new LiveSignalsBridge;

    init();
    setContextMenuPolicy(Qt::NoContextMenu);

//#ifdef __linux__
    if (char *p = zeno::envconfig::get("OPEN")) {
        zeno::log_info("ZENO_OPEN: {}", p);
        openFile(p);
    }
//#endif
    m_spCacheMgr = std::make_shared<ZCacheMgr>();
}

ZenoMainWindow::~ZenoMainWindow()
{
    delete liveTcpServer;
    delete liveHttpServer;
    delete liveSignalsBridge;
}

void ZenoMainWindow::init()
{
    m_ui = new Ui::MainWindow;
    m_ui->setupUi(this);

    initMenu();
    initLive();
    initDocks();
    initWindowProperty();

    addToolBar(Qt::LeftToolBarArea, new FakeToolbar(false));
    addToolBar(Qt::RightToolBarArea, new FakeToolbar(false));
    addToolBar(Qt::BottomToolBarArea, new FakeToolbar(true));
    addToolBar(Qt::TopToolBarArea, new FakeToolbar(true));

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(11, 11, 11));
    setAutoFillBackground(true);
    setPalette(pal);

    m_ui->statusbar->showMessage(tr("Status Bar"));
}

void ZenoMainWindow::initWindowProperty()
{
    auto pGraphsMgm = zenoApp->graphsManagment();
    setWindowIcon(QIcon(":/icons/zeno-logo.png"));
    setWindowTitle(UiHelper::nativeWindowTitle(""));
    connect(pGraphsMgm, &GraphsManagment::fileOpened, this, [=](QString fn) {
        QFileInfo info(fn);
        QString path = info.filePath();
        QString title = UiHelper::nativeWindowTitle(path);
        updateNativeWinTitle(title);
    });
    connect(pGraphsMgm, &GraphsManagment::fileClosed, this, [=]() { 
        QString title = UiHelper::nativeWindowTitle("");
        updateNativeWinTitle(title);
    });
    connect(pGraphsMgm, &GraphsManagment::fileSaved, this, [=](QString fn) {
        QFileInfo info(fn);
        QString path = info.filePath();
        QString title = UiHelper::nativeWindowTitle(path);
        updateNativeWinTitle(title);
    });
}

void ZenoMainWindow::updateNativeWinTitle(const QString& title)
{
    QWidgetList lst = QApplication::topLevelWidgets();
    for (auto wid : lst)
    {
        if (qobject_cast<ZTabDockWidget*>(wid) ||
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

    QActionGroup *actionGroup = new QActionGroup(this);
    actionGroup->addAction(m_ui->actionShading);
    actionGroup->addAction(m_ui->actionSolid);
    actionGroup->addAction(m_ui->actionOptix);
    m_ui->actionSolid->setChecked(true);

    auto actions = findChildren<QAction*>(QString(), Qt::FindDirectChildrenOnly);
    for (QAction* action : actions)
    {
        connect(action, SIGNAL(triggered(bool)), this, SLOT(onMenuActionTriggered(bool)));  
        setActionIcon(action);
    }

    m_ui->menubar->setProperty("cssClass", "mainWin");
    //qt bug: qss font is not valid on menubar.
    QFont font = zenoApp->font();
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
}

void ZenoMainWindow::onMenuActionTriggered(bool bTriggered)
{
    QAction* pAction = qobject_cast<QAction*>(sender());
    int actionType = pAction->property("ActionType").toInt();
    if (actionType == ACTION_SHADING || actionType == ACTION_SOLID || actionType == ACTION_OPTIX) 
    {
        setActionIcon(m_ui->actionShading);
        setActionIcon(m_ui->actionSolid);
        setActionIcon(m_ui->actionOptix);
    } 
    else 
    {
        setActionIcon(pAction);
    }
    switch (actionType)
    {
    case ACTION_NEW: {
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
        saveQuit();
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
    auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    DisplayWidget* pViewport = nullptr;
    ZenoGraphsEditor* pEditor = nullptr;
    for (ZTabDockWidget* pDock : docks)
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
    auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget *pDock : docks) {
        pDock->close();
        //pDock->testCleanupGL();
        //delete pDock;
    }

    m_layoutRoot = root;
    ZTabDockWidget* cake = new ZTabDockWidget(this);
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

void ZenoMainWindow::initDocksWidget(ZTabDockWidget* pLeft, PtrLayoutNode root)
{
    if (!root)
        return;

    if (root->type == NT_HOR || root->type == NT_VERT)
    {
        ZTabDockWidget* pRight = new ZTabDockWidget(this);
        Qt::Orientation ori = root->type == NT_HOR ? Qt::Horizontal : Qt::Vertical;
        splitDockWidget(pLeft, pRight, ori);
        initDocksWidget(pLeft, root->pLeft);
        initDocksWidget(pRight, root->pRight);
    }
    else if (root->type == NT_ELEM)
    {
        root->pWidget = pLeft;
        for (QString tab : root->tabs)
        {
            PANEL_TYPE type = ZTabDockWidget::title2Type(tab);
            if (type != PANEL_EMPTY)
            {
                pLeft->onAddTab(type);
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

void ZenoMainWindow::initDocks() {
    /*m_layoutRoot = std::make_shared<LayerOutNode>();
    m_layoutRoot->type = NT_ELEM;

    ZTabDockWidget* viewDock = new ZTabDockWidget(this);
    viewDock->setCurrentWidget(PANEL_VIEW);
    viewDock->setObjectName("viewDock");

    ZTabDockWidget *logDock = new ZTabDockWidget(this);
    logDock->setCurrentWidget(PANEL_LOG);
    logDock->setObjectName("logDock");

    ZTabDockWidget *paramDock = new ZTabDockWidget(this);
    paramDock->setCurrentWidget(PANEL_NODE_PARAMS);
    paramDock->setObjectName("paramDock");

    ZTabDockWidget* editorDock = new ZTabDockWidget(this);
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

    connect(m_pTimeline, &ZTimeline::sliderValueChanged, this, [=](int frame) {
        QVector<DisplayWidget*> views = viewports();
        for (DisplayWidget* view : views) {
            view->onSliderValueChanged(frame);
        }
    });

    auto graphs = zenoApp->graphsManagment();
    connect(graphs, &GraphsManagment::modelDataChanged, this, [=]() {
        if (m_bAlways) {
            killProgram();
        }
    });
}

ZTimeline* ZenoMainWindow::timeline() const
{
    return m_pTimeline;
}

void ZenoMainWindow::onMaximumTriggered()
{
    ZTabDockWidget* pDockWidget = qobject_cast<ZTabDockWidget*>(sender());
    auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* pDock : docks)
    {
        if (pDock != pDockWidget)
        {
            pDock->close();
        }
    }
}

QVector<DisplayWidget*> ZenoMainWindow::viewports() const
{
    QVector<DisplayWidget*> views;
    auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* pDock : docks)
    {
        if (pDock->isVisible())
            views.append(pDock->viewports());
    }

    //top level floating windows.
    QWidgetList lst = QApplication::topLevelWidgets();
    for (auto wid : lst)
    {
        if (ZTabDockWidget* pFloatWin = qobject_cast<ZTabDockWidget*>(wid))
        {
            views.append(pFloatWin->viewports());
        }
    }
    return views;
}

void ZenoMainWindow::toggleTimelinePlay(bool bOn)
{
    m_pTimeline->togglePlayButton(bOn);
}

void ZenoMainWindow::onRunTriggered()
{
    QVector<DisplayWidget*> views = viewports();

    clearErrorMark();

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
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel* pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        launchProgram(pModel, beginFrame, endFrame);
    }

    for (auto view : views)
    {
        view->afterRun();
    }
}

void ZenoMainWindow::directlyRunRecord(const ZENO_RECORD_RUN_INITPARAM& param)
{
#if 0
    ZASSERT_EXIT(m_viewDock);
    DisplayWidget* viewWidget = qobject_cast<DisplayWidget *>(m_viewDock->widget());
    ZASSERT_EXIT(viewWidget);
    ViewportWidget* pViewport = viewWidget->getViewportWidget();
    ZASSERT_EXIT(pViewport);

    //hide other component
    if (m_editor) m_editor->hide();
    if (m_logger) m_logger->hide();
    if (m_parameter) m_parameter->hide();

    VideoRecInfo recInfo;
    recInfo.bitrate = param.iBitrate;
    recInfo.fps = param.iFps;
    recInfo.frameRange = {param.iSFrame, param.iSFrame + param.iFrame - 1};
    recInfo.numMSAA = 0;
    recInfo.numOptix = 1;
    recInfo.numSamples = param.iSample;
    recInfo.audioPath = param.audioPath;
    recInfo.record_path = param.sPath;
    recInfo.bRecordRun = true;
    recInfo.videoname = "output.mp4";
    recInfo.exitWhenRecordFinish = param.exitWhenRecordFinish;

    if (!param.sPixel.isEmpty())
    {
        QStringList tmpsPix = param.sPixel.split("x");
        int pixw = tmpsPix.at(0).toInt();
        int pixh = tmpsPix.at(1).toInt();
        recInfo.res = {(float)pixw, (float)pixh};

        pViewport->setFixedSize(pixw, pixh);
        pViewport->setCameraRes(QVector2D(pixw, pixh));
        pViewport->updatePerspective();
    } else {
        recInfo.res = {(float)1000, (float)680};
        pViewport->setMinimumSize(1000, 680);
    }

    auto sess = Zenovis::GetInstance().getSession();
    if (sess) {
        auto scene = sess->get_scene();
        if (scene) {
            scene->drawOptions->num_samples = param.bRecord ? param.iSample : 16;
        }
    }

    bool ret = openFile(param.sZsgPath);
    ZASSERT_EXIT(ret);
    viewWidget->runAndRecord(recInfo);
#endif
}

void ZenoMainWindow::updateViewport(const QString& action)
{
    QVector<DisplayWidget*> views = viewports();
    for (DisplayWidget* view : views)
    {
        view->updateFrame(action);
    }
}

ZenoGraphsEditor* ZenoMainWindow::getAnyEditor() const
{
    auto docks2 = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
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
    auto docks2 = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto dock : docks2)
    {
        dock->onFinished();
    }
}

void ZenoMainWindow::onCloseDock()
{
    ZTabDockWidget *pDockWidget = qobject_cast<ZTabDockWidget *>(sender());
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

void ZenoMainWindow::SplitDockWidget(ZTabDockWidget* after, ZTabDockWidget* dockwidget, Qt::Orientation orientation)
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
    ZTabDockWidget* pDockWidget = qobject_cast<ZTabDockWidget*>(sender());
    ZTabDockWidget* pDock = new ZTabDockWidget(this);

    //QLayout* pLayout = this->layout();
    //QMainWindowLayout* pWinLayout = qobject_cast<QMainWindowLayout*>(pLayout);

    pDock->setObjectName("editorDock233");
    pDock->setCurrentWidget(PANEL_EDITOR);
    //pDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    SplitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
}

void ZenoMainWindow::openFileDialog()
{
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;

    //todo: path validation
    if (saveQuit()) 
    {
        openFile(filePath);
    }
}

void ZenoMainWindow::onNewFile() {
    if (saveQuit()) 
    {
        zenoApp->graphsManagment()->newFile();
    }
}

void ZenoMainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
}

std::shared_ptr<ZCacheMgr> ZenoMainWindow::cacheMgr() const
{
    return m_spCacheMgr;
}

void ZenoMainWindow::closeEvent(QCloseEvent *event)
{
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

        auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget *pDock : docks) {
            pDock->close();
            try {
                pDock->testCleanupGL();
            } catch (...) {
                //QString errMsg = QString::fromLatin1(e.what());
                int j;
                j = 0;
            }
            delete pDock;
        }

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
    return QMainWindow::event(event);
}

void ZenoMainWindow::importGraph() {
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;

    //todo: path validation
    auto pGraphs = zenoApp->graphsManagment();
    pGraphs->importGraph(filePath);
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

void ZenoMainWindow::exportGraph() {
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Export", "",
                                                "C++ Source File(*.cpp);; JSON file(*.json);; All Files(*);;");
    if (path.isEmpty()) {
        return;
    }

    //auto pGraphs = zenoApp->graphsManagment();
    //pGraphs->importGraph(path);

    QString content;
    {
        IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
        if (path.endsWith(".cpp")) {
            content = serializeSceneCpp(pModel);
        } else {
            rapidjson::StringBuffer s;
            RAPIDJSON_WRITER writer(s);
            writer.StartArray();
            serializeScene(pModel, writer);
            writer.EndArray();
            content = QString(s.GetString());
        }
    }
    saveContent(content, path);
}

bool ZenoMainWindow::openFile(QString filePath)
{
    auto pGraphs = zenoApp->graphsManagment();
    IGraphsModel* pModel = pGraphs->openZsgFile(filePath);
    if (!pModel)
        return false;

    resetTimeline(pGraphs->timeInfo());
    recordRecentFile(filePath);
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
                bool ret = openFile(path);
                if (!ret) {
                    int flag = QMessageBox::question(nullptr, "", tr("the file does not exies, do you want to remove it?"), QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                    if (flag & QMessageBox::Yes) {
                        QSettings _settings(QSettings::UserScope, zsCompanyName, zsEditor);
                        _settings.beginGroup("Recent File List");
                        _settings.remove(key);
                        m_ui->menuRecent_Files->removeAction(action);
                    }
                }
            });
        }
    }
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
    emit recentFilesChanged();
}

void ZenoMainWindow::setActionProperty() 
{
    m_ui->action_New->setProperty("ActionType", ACTION_NEW);
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
    m_ui->actionSmooth_Shading->setProperty("ActionType", ACTION_SMOOTH_SHADING);
    m_ui->actionNormal_Check->setProperty("ActionType", ACTION_NORMAL_CHECK);
    m_ui->actionWireFrame->setProperty("ActionType", ACTION_WIRE_FRAME);
    m_ui->actionShow_Grid->setProperty("ActionType", ACTION_SHOW_GRID);
    m_ui->actionBackground_Color->setProperty("ActionType", ACTION_BACKGROUND_COLOR);
    m_ui->actionSolid->setProperty("ActionType", ACTION_SOLID);
    m_ui->actionShading->setProperty("ActionType", ACTION_SHADING);
    m_ui->actionOptix->setProperty("ActionType", ACTION_OPTIX);
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
            ViewportWidget* pViewport = pWid->getViewportWidget();
            ZASSERT_EXIT(pViewport);
            pViewport->getSession()->do_screenshot(path.toStdString(), ext.toStdString());
        }
    }
}

void ZenoMainWindow::setActionIcon(QAction *action) 
{
    if (!action->isCheckable() || !action->isChecked()) 
    {
        action->setIcon(QIcon());
    }
    if (action->isChecked()) 
    {
        action->setIcon(QIcon("://icons/checked.png"));
    }
}

bool ZenoMainWindow::saveQuit() {
    auto pGraphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgm, true);
    IGraphsModel *pModel = pGraphsMgm->currentModel();
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
    resetTimeline(TIMELINE_INFO());
    return true;
}

void ZenoMainWindow::save()
{
    auto pGraphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgm);
    IGraphsModel* pModel = pGraphsMgm->currentModel();
    zenoio::ZSG_VERSION ver = pModel->ioVersion();
    if (zenoio::VER_2 == ver)
    {
        QMessageBox msgBox(QMessageBox::Information, "", QString::fromLocal8Bit("当前zsg为旧格式文件，为了确保不被新格式覆盖，只能通过“另存为”操作保存为新格式"));
        msgBox.exec();
        bool ret = saveAs();
        if (ret) {
            pModel->setIOVersion(zenoio::VER_2_5);
        }
    }
    else
    {
        if (pModel)
        {
            QString currFilePath = pModel->filePath();
            if (currFilePath.isEmpty())
                saveAs();
            else
                saveFile(currFilePath);
        }
    }
}

bool ZenoMainWindow::saveFile(QString filePath)
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    APP_SETTINGS settings;
    settings.timeline = timelineInfo();
    zenoApp->graphsManagment()->saveFile(filePath, settings);
    recordRecentFile(filePath);
    return true;
}

bool ZenoMainWindow::inDlgEventLoop() const {
    return m_bInDlgEventloop;
}

void ZenoMainWindow::setInDlgEventLoop(bool bOn) {
    m_bInDlgEventloop = bOn;
}

TIMELINE_INFO ZenoMainWindow::timelineInfo()
{
    TIMELINE_INFO info;
    ZASSERT_EXIT(m_pTimeline, info);
    info.bAlways = m_bAlways;
    info.beginFrame = m_pTimeline->fromTo().first;
    info.endFrame = m_pTimeline->fromTo().second;
    return info;
}

bool ZenoMainWindow::isAlways() const
{
    return m_bAlways;
}

void ZenoMainWindow::setAlways(bool bAlways)
{
    m_bAlways = bAlways;
    emit alwaysModeChanged(bAlways);
}

void ZenoMainWindow::resetTimeline(TIMELINE_INFO info)
{
    setAlways(info.bAlways);
    m_pTimeline->initFromTo(info.beginFrame, info.endFrame);
}

void ZenoMainWindow::onFeedBack()
{
    /*
    ZFeedBackDlg dlg(this);
    if (dlg.exec() == QDialog::Accepted)
    {
        QString content = dlg.content();
        bool isSend = dlg.isSendFile();
        if (isSend)
        {
            IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
            if (!pModel) {
                return;
            }
            QString strContent = ZsgWriter::getInstance().dumpProgramStr(pModel);
            dlg.sendEmail("bug feedback", content, strContent);
        }
    }
    */
}

void ZenoMainWindow::clearErrorMark()
{
    //clear all error mark at every scene.
    auto graphsMgm = zenoApp->graphsManagment();
    IGraphsModel* pModel = graphsMgm->currentModel();
    if (!pModel) {
        return;
    }
    const QModelIndexList& lst = pModel->subgraphsIndice();
    for (const QModelIndex& idx : lst)
    {
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(graphsMgm->gvScene(idx));
        if (!pScene) {
            pScene = new ZenoSubGraphScene(graphsMgm);
            graphsMgm->addScene(idx, pScene);
            pScene->initModel(idx);
        }

        if (pScene) {
            pScene->clearMark();
        }
    }
}

bool ZenoMainWindow::saveAs() {
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zeno Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty()) {
        return saveFile(path);
    }
    return false;
}

QString ZenoMainWindow::getOpenFileByDialog() {
    DlgInEventLoopScope;
    const QString &initialPath = "";
    QFileDialog fileDialog(this, tr("Open"), initialPath, "Zeno Graph File (*.zsg)\nAll Files (*)");
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setFileMode(QFileDialog::ExistingFile);
    if (fileDialog.exec() != QDialog::Accepted)
        return "";

    QString filePath = fileDialog.selectedFiles().first();
    return filePath;
}

void ZenoMainWindow::onNodesSelected(const QModelIndex &subgIdx, const QModelIndexList &nodes, bool select) {
    //dispatch to all property panel.
    auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->onNodesSelected(subgIdx, nodes, select);
    }
}

void ZenoMainWindow::onPrimitiveSelected(const std::unordered_set<std::string>& primids) {
    //dispatch to all property panel.
    auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->onPrimitiveSelected(primids);
    }
}

void ZenoMainWindow::updateLightList() {
    auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* dock : docks) {
        if (dock->isVisible())
            dock->newFrameUpdate();
    }
}
void ZenoMainWindow::doFrameUpdate(int frame) {
    if(liveHttpServer->clients.empty())
        return;

    std::cout << "====== Frame " << frame << "\n";

    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    for (auto displayWid : views)
    {
        ZASSERT_EXIT(displayWid);
        auto viewport = displayWid->getViewportWidget();
        ZASSERT_EXIT(viewport);
        std::cout << "====== CameraMoving " << viewport->m_bMovingCamera << "\n";

        // Sync Camera
        if(viewport->m_bMovingCamera){

        }
        // Sync Frame
        else {
            int count = liveHttpServer->frameMeshDataCount(frame);
            std::string data = "FRAME " + std::to_string(frame) + " SYNCMESH " + std::to_string(count);
            for(auto& c: liveHttpServer->clients) {
                auto r = liveTcpServer->sendData({c.first, c.second, data});
                std::cout << "\tClient " << c.first << ":" << c.second << " Receive " << r.data << "\n";
            }
        }
    }
}
