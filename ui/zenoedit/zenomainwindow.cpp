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
#include <zenomodel/include/modeldata.h>
#include <zenoui/style/zenostyle.h>
#include <zenomodel/include/uihelper.h>
#include "util/log.h"
#include "dialog/zfeedbackdlg.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"
#include "panel/zenolights.h"
#include "nodesys/zenosubgraphscene.h"
#include "ui_zenomainwindow.h"


ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , m_bInDlgEventloop(false)
    , m_pTimeline(nullptr)
    , m_layoutRoot(nullptr)
    , m_nResizeTimes(0)
{
    init();
    setContextMenuPolicy(Qt::NoContextMenu);
    setWindowTitle("Zeno Editor (" + QString::fromStdString(getZenoVersion()) + ")");
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

void ZenoMainWindow::init()
{
    m_ui = new Ui::MainWindow;
    m_ui->setupUi(this);

    initMenu();
    initDocks();

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(11, 11, 11));
    setAutoFillBackground(true);
    setPalette(pal);

    m_ui->statusbar->showMessage(tr("Status Bar"));
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
    auto actions = findChildren<QAction*>(QString(), Qt::FindDirectChildrenOnly);
    for (QAction* action : actions)
    {
        connect(action, SIGNAL(triggered(bool)), this, SLOT(onMenuActionTriggered(bool)));
        if (!action->isCheckable())
            action->setIcon(QIcon());
    }

    connect(m_ui->action_New, SIGNAL(triggered()), this, SLOT(onNewFile()));
    connect(m_ui->action_Open, SIGNAL(triggered()), this, SLOT(openFileDialog()));
    connect(m_ui->action_Save, SIGNAL(triggered()), this, SLOT(save()));
    connect(m_ui->action_Save_As, SIGNAL(triggered()), this, SLOT(saveAs()));
    connect(m_ui->action_Import, SIGNAL(triggered()), this, SLOT(importGraph()));
    connect(m_ui->actionExportGraph, SIGNAL(triggered()), this, SLOT(exportGraph()));
    connect(m_ui->action_Close, SIGNAL(triggered()), this, SLOT(saveQuit()));
    connect(m_ui->actionSave_Layout, SIGNAL(triggered()), this, SLOT(saveDockLayout()));
    connect(m_ui->actionEnglish_Chinese, SIGNAL(triggered(bool)), this, SLOT(onLangChanged(bool)));

    m_ui->menubar->setProperty("cssClass", "mainWin");

    //check user saved layout.
    loadSavedLayout();
}

void ZenoMainWindow::onMenuActionTriggered(bool bTriggered)
{
    QAction* pAction = qobject_cast<QAction*>(sender());
    dispatchCommand(pAction, bTriggered);
}

void ZenoMainWindow::dispatchCommand(QAction* pAction, bool bTriggered)
{
    if (!pAction)
        return;

    //dispatch to every panel.
    auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* pDock : docks)
    {
        pDock->onMenuActionTriggered(pAction, bTriggered);
    }
}

void ZenoMainWindow::loadSavedLayout()
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("layout");
    QStringList lst = settings.childGroups();
    if (!lst.isEmpty())
    {
        for (QString name : lst)
        {
            QAction* pCustomLayout_ = new QAction(name);
            connect(pCustomLayout_, &QAction::triggered, this, [=]() {
                QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
                settings.beginGroup("layout");
                settings.beginGroup(name);
                if (settings.allKeys().indexOf("content") != -1)
                {
                    QString content = settings.value("content").toString();
                    PtrLayoutNode root = readLayout(content);
                    resetDocks(root);
                }
                else {
                    QMessageBox msg(QMessageBox::Warning, "", tr("layout format is invalid."));
                    msg.exec();
                }
                settings.endGroup();
                settings.endGroup();
                });
            m_ui->menuCustom_Layout->addAction(pCustomLayout_);
        }
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
    }
}

void ZenoMainWindow::saveLayout2()
{
    auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    QLayout* pLayout = this->layout();
    //QMainWindowLayout* pWinLayout = qobject_cast<QMainWindowLayout*>(pLayout);
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "JSON file(*.json);;");
    writeLayout(m_layoutRoot, size(), path);
}

void ZenoMainWindow::onLangChanged(bool bChecked)
{
    QSettings settings(zsCompanyName, zsEditor);
    settings.setValue("use_chinese", bChecked);
    QMessageBox msg(QMessageBox::Information, "Language",
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
        delete pDock;
    }

    m_layoutRoot = root;
    ZTabDockWidget *cake = new ZTabDockWidget(this);
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

void ZenoMainWindow::initDocks()
{
    m_layoutRoot = std::make_shared<LayerOutNode>();
    m_layoutRoot->type = NT_ELEM;

    ZTabDockWidget* viewDock = new ZTabDockWidget(this);
    viewDock->setCurrentWidget(PANEL_VIEW);
    viewDock->setObjectName("viewDock");

    ZTabDockWidget *logDock = new ZTabDockWidget(this);
    logDock->setCurrentWidget(PANEL_LOG);
    logDock->setObjectName("logDock");

    ZTabDockWidget* paramDock = new ZTabDockWidget(this);
    paramDock->setCurrentWidget(PANEL_NODE_PARAMS);
    paramDock->setObjectName("paramDock");

    ZTabDockWidget* editorDock = new ZTabDockWidget(this);
    editorDock->setCurrentWidget(PANEL_EDITOR);
    editorDock->setObjectName("editorDock");

    addDockWidget(Qt::TopDockWidgetArea, viewDock);
    initTimelineDock();
    m_layoutRoot->type = NT_ELEM;
    m_layoutRoot->pWidget = viewDock;

    SplitDockWidget(viewDock, editorDock, Qt::Vertical);
    SplitDockWidget(viewDock, logDock, Qt::Horizontal);
    SplitDockWidget(editorDock, paramDock, Qt::Horizontal);
}

void ZenoMainWindow::initTimelineDock()
{
    m_pTimeline = new ZTimeline;
    setCentralWidget(m_pTimeline);

    connect(m_pTimeline, &ZTimeline::playForward, this, [=](bool bPlaying) {
        auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget* pDock : docks)
            pDock->onPlayClicked(bPlaying);
    });

    connect(m_pTimeline, &ZTimeline::sliderValueChanged, this, [=](int frame) {
        auto docks = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget* pDock : docks)
            pDock->onSliderValueChanged(frame);
    });

    connect(m_pTimeline, &ZTimeline::run, this, [=]() {
        auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget *pDock : docks)
            pDock->onRun();
    });

    connect(m_pTimeline, &ZTimeline::kill, this, [=]() {
        auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget *pDock : docks)
            pDock->onKill();
    });

    connect(m_pTimeline, &ZTimeline::alwaysChecked, this, [=]() {
        auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
        for (ZTabDockWidget *pDock : docks)
            pDock->onRun();
    });

    auto graphs = zenoApp->graphsManagment();
    connect(graphs, &GraphsManagment::modelDataChanged, this, [=]() {
        if (m_pTimeline->isAlways()) {
            auto docks = findChildren<ZTabDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
            for (ZTabDockWidget *pDock : docks)
                pDock->onRun();
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

void ZenoMainWindow::updateViewport(const QString& action)
{
    auto docks2 = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto dock : docks2)
    {
        dock->onUpdateViewport(action);
    }
}

DisplayWidget* ZenoMainWindow::getDisplayWidget()
{
    //DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    //if (view)
    //    return view;
    return nullptr;
}

void ZenoMainWindow::onRunFinished()
{
    auto docks2 = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (auto dock : docks2)
    {
        dock->onRunFinished();
    }
}

void ZenoMainWindow::onCloseDock()
{
    ZTabDockWidget *pDockWidget = qobject_cast<ZTabDockWidget *>(sender());
    ZASSERT_EXIT(pDockWidget);
    pDockWidget->close();

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
    saveQuit();
    openFile(filePath);
}

void ZenoMainWindow::onNewFile() {
    saveQuit();
    zenoApp->graphsManagment()->newFile();
}

void ZenoMainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
}

void ZenoMainWindow::closeEvent(QCloseEvent *event)
{
    this->saveQuit();
    // todo: event->ignore() when saveQuit returns false?
    QMainWindow::closeEvent(event);
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
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "",
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

void ZenoMainWindow::recordRecentFile(const QString& filePath)
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");

    QStringList keys = settings.childKeys();
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
        QString fn = keys[keys.length() - 1];
        static QRegExp rx("File (\\d+)");
        if (rx.indexIn(fn) != -1) {
            QStringList caps = rx.capturedTexts();
            if (caps.length() == 2)
                idx = caps[1].toInt();
        } else {
            //todo
        }
    }

    settings.setValue(QString("File %1").arg(idx + 1), filePath);
}

void ZenoMainWindow::onToggleDockWidget(DOCK_TYPE type, bool bShow)
{
    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *dock : docks)
    {
        DOCK_TYPE _type = dock->type();
        if (_type == type)
            dock->setVisible(bShow);
    }
}

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
}

void ZenoMainWindow::onDockSwitched(DOCK_TYPE type)
{
    ZenoDockWidget *pDock = qobject_cast<ZenoDockWidget *>(sender());
    switch (type)
    {
        case DOCK_EDITOR: {
            ZenoGraphsEditor *pEditor2 = new ZenoGraphsEditor(this);
            pEditor2->resetModel(zenoApp->graphsManagment()->currentModel());
            pDock->setWidget(type, pEditor2);
            break;
        }
        case DOCK_VIEW: {
            //complicated opengl framework.
            DisplayWidget* view = new DisplayWidget;
            pDock->setWidget(type, view);
            break;
        }
        case DOCK_NODE_PARAMS: {
            ZenoPropPanel *pWidget = new ZenoPropPanel;
            pDock->setWidget(type, pWidget);
            break;
        }
        case DOCK_NODE_DATA: {
            ZenoSpreadsheet *pWidget = new ZenoSpreadsheet;
            pDock->setWidget(type, pWidget);
            break;
        }
        case DOCK_LOG: {
            ZPlainLogPanel* pPanel = new ZPlainLogPanel;
            pDock->setWidget(type, pPanel);
            break;
        }
        case DOCK_LIGHTS: {
            ZenoLights* pPanel = new ZenoLights;
            pDock->setWidget(type, pPanel);
            break;
        }
    }
    pDock->setObjectName(uniqueDockObjName(type));
}

void ZenoMainWindow::saveQuit() {
    auto pGraphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgm);
    IGraphsModel *pModel = pGraphsMgm->currentModel();
    if (!zeno::envconfig::get("OPEN") /* <- don't annoy me when I'm debugging via ZENO_OPEN */ && pModel && pModel->isDirty()) {
        QMessageBox msgBox(QMessageBox::Question, tr("Save"), tr("Save changes?"), QMessageBox::Yes | QMessageBox::No, this);
        QPalette pal = msgBox.palette();
        pal.setBrush(QPalette::WindowText, QColor(0, 0, 0));
        msgBox.setPalette(pal);
        int ret = msgBox.exec();
        if (ret & QMessageBox::Yes) {
            save();
        }
    }
    pGraphsMgm->clear();
    //clear timeline info.
    resetTimeline(TIMELINE_INFO());
}

void ZenoMainWindow::save()
{
    auto pGraphsMgm = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgm);
    IGraphsModel *pModel = pGraphsMgm->currentModel();
    if (pModel) {
        QString currFilePath = pModel->filePath();
        if (currFilePath.isEmpty())
            return saveAs();
        saveFile(currFilePath);
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
    info.bAlways = m_pTimeline->isAlways();
    info.beginFrame = m_pTimeline->fromTo().first;
    info.endFrame = m_pTimeline->fromTo().second;
    return info;
}

void ZenoMainWindow::resetTimeline(TIMELINE_INFO info)
{
    m_pTimeline->setAlways(info.bAlways);
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

void ZenoMainWindow::saveAs() {
    DlgInEventLoopScope;
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zeno Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty()) {
        saveFile(path);
    }
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
    auto docks2 = findChildren<ZTabDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZTabDockWidget* dock : docks2) {
        dock->onNodesSelected(subgIdx, nodes, select);
    }
}

void ZenoMainWindow::onPrimitiveSelected(const std::unordered_set<std::string>& primids) {
    //dispatch to all property panel.
    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *dock : docks) {
        dock->onPrimitiveSelected(primids);
    }
}

void ZenoMainWindow::updateLightList() {
    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *dock : docks) {
        dock->newFrameUpdate();
    }
}
