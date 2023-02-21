#include "launch/livehttpserver.h"
#include "launch/livetcpserver.h"
#include "zenomainwindow.h"
#include "dock/zenodockwidget.h"
#include <zenomodel/include/graphsmanagment.h>
#include "launch/corelaunch.h"
#include "launch/serialize.h"
#include "nodesview/zenographseditor.h"
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


ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , m_pEditor(nullptr)
    , m_viewDock(nullptr)
    , m_bInDlgEventloop(false)
    , m_logger(nullptr)
{
    liveTcpServer = new LiveTcpServer;
    liveHttpServer = new LiveHttpServer;
    liveSignalsBridge = new LiveSignalsBridge;

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
    delete liveTcpServer;
    delete liveHttpServer;
    delete liveSignalsBridge;
}

void ZenoMainWindow::init()
{
    initMenu();
    initLive();
    initDocks();
    verticalLayout();
    //onlyEditorLayout();

    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(11, 11, 11));
    setAutoFillBackground(true);
    setPalette(pal);
}

void ZenoMainWindow::initLive() {

}

void ZenoMainWindow::initMenu() {
    QMenuBar *pMenuBar = new QMenuBar(this);
    if (!pMenuBar)
        return;

    QMenu *pFile = new QMenu(tr("File"));
    {
        QAction *pAction = new QAction(tr("New"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+N")));
        pAction->setShortcutContext(Qt::ApplicationShortcut);
        //QMenu *pNewMenu = new QMenu;
        //QAction *pNewGraph = pNewMenu->addAction("New Scene");
        connect(pAction, SIGNAL(triggered()), this, SLOT(onNewFile()));

        //pAction->setMenu(pNewMenu);

        pFile->addAction(pAction);

        pAction = new QAction(tr("Open"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+O")));
        pAction->setShortcutContext(Qt::ApplicationShortcut);
        connect(pAction, SIGNAL(triggered()), this, SLOT(openFileDialog()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Save"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+S")));
        pAction->setShortcutContext(Qt::ApplicationShortcut);
        connect(pAction, SIGNAL(triggered()), this, SLOT(save()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Save As"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+Shift+S")));
        pAction->setShortcutContext(Qt::ApplicationShortcut);
        connect(pAction, SIGNAL(triggered()), this, SLOT(saveAs()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Import"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+Shift+O")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(importGraph()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Export"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(("Ctrl+Shift+E")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(exportGraph()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Close"), pFile);
        connect(pAction, SIGNAL(triggered()), this, SLOT(saveQuit()));
        pFile->addAction(pAction);
    }

    QMenu *pEdit = new QMenu(tr("Edit"));
    {
        QAction *pAction = new QAction(tr("Undo"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Redo"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Cut"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Copy"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Paste"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);
    }

    //QMenu *pRender = new QMenu(tr("Render"));

    QMenu *pView = new QMenu(tr("View"));
    {
        QAction *pViewAct = new QAction(tr("view"));
        pViewAct->setCheckable(true);
        pViewAct->setChecked(true);
        connect(pViewAct, &QAction::triggered, this, [=]() { onToggleDockWidget(DOCK_VIEW, pViewAct->isChecked()); });
        pView->addAction(pViewAct);

        QAction* pEditorAct = new QAction(tr("editor"));
        pEditorAct->setCheckable(true);
        pEditorAct->setChecked(true);
        connect(pEditorAct, &QAction::triggered, this, [=]() { onToggleDockWidget(DOCK_EDITOR, pEditorAct->isChecked()); });
        pView->addAction(pEditorAct);

        QAction* pPropAct = new QAction(tr("property"));
        pPropAct->setCheckable(true);
        pPropAct->setChecked(true);
        connect(pPropAct, &QAction::triggered, this, [=]() { onToggleDockWidget(DOCK_NODE_PARAMS, pPropAct->isChecked()); });
        pView->addAction(pPropAct);

        QAction* pLoggerAct = new QAction(tr("logger"));
        pLoggerAct->setCheckable(true);
        pLoggerAct->setChecked(true);
        connect(pLoggerAct, &QAction::triggered, this, [=]() { onToggleDockWidget(DOCK_LOG, pLoggerAct->isChecked()); });
        pView->addAction(pLoggerAct);

        connect(pView, &QMenu::aboutToShow, this, [=]() {
            auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
            for (ZenoDockWidget *dock : docks)
            {
                DOCK_TYPE type = dock->type();
                switch (type)
                {
                    case DOCK_VIEW:     pViewAct->setChecked(dock->isVisible());    break;
                    case DOCK_EDITOR:   pEditorAct->setChecked(dock->isVisible());  break;
                    case DOCK_NODE_PARAMS: pPropAct->setChecked(dock->isVisible()); break;
                    case DOCK_LOG:      pLoggerAct->setChecked(dock->isVisible()); break;
                }
            }
        });

        pView->addSeparator();

        QAction* pSaveLayout = new QAction(tr("Save Layout"));
        connect(pSaveLayout, &QAction::triggered, this, [=]() {
            bool bOk = false;
            QString name = QInputDialog::getText(this, tr("Save Layout"), tr("layout name:"),
                                                        QLineEdit::Normal, "layout_1", &bOk);
            if (bOk) {
                QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
                settings.beginGroup("layout");
                if (settings.childGroups().indexOf(name) != -1) {
                    QMessageBox msg(QMessageBox::Warning, "", tr("alreday has same layout"));
                    msg.exec();
                    settings.endGroup();
                    return;
                }

                settings.beginGroup(name);
                settings.setValue("geometry", saveGeometry());
                settings.setValue("state", saveState());
                settings.endGroup();
                settings.endGroup();
            }
        });
        pView->addAction(pSaveLayout);

        //check user saved layout.
        QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
        settings.beginGroup("layout");
        QStringList lst = settings.childGroups();
        if (!lst.isEmpty())
        {
            QMenu* pCustomLayout = new QMenu(tr("Custom Layout"));

            for (QString name : lst)
            {
                QAction *pCustomLayout_ = new QAction(name);
                connect(pCustomLayout_, &QAction::triggered, this, [=]() {
                    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
                    settings.beginGroup("layout");
                    settings.beginGroup(name);
                    restoreGeometry(settings.value("geometry").toByteArray());
                    restoreState(settings.value("state").toByteArray());
                    settings.endGroup();
                    settings.endGroup();
                });
                pCustomLayout->addAction(pCustomLayout_);
            }
            pView->addMenu(pCustomLayout);
        }
    }

    //QMenu *pWindow = new QMenu(tr("Window"));

    QMenu *pHelp = new QMenu(tr("Help"));
    {
        QAction *pAction = new QAction(tr("Send this File"));
        connect(pAction, SIGNAL(triggered(bool)), this, SLOT(onFeedBack()));
        pHelp->addAction(pAction);

        pHelp->addSeparator();

        pAction = new QAction(tr("English / Chinese"), this);
        pAction->setCheckable(true);
        {
            QSettings settings(zsCompanyName, zsEditor);
            QVariant use_chinese = settings.value("use_chinese");
            pAction->setChecked(use_chinese.isNull() || use_chinese.toBool());
        }
        pHelp->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            QSettings settings(zsCompanyName, zsEditor);
            settings.setValue("use_chinese", pAction->isChecked());
            QMessageBox msg(QMessageBox::Information, "Language",
                        tr("Please restart Zeno to apply changes."),
                        QMessageBox::Ok, this);
            msg.exec();
        });
    }

    pMenuBar->addMenu(pFile);
    pMenuBar->addMenu(pEdit);
    //pMenuBar->addMenu(pRender);
    pMenuBar->addMenu(pView);
    //pMenuBar->addMenu(pWindow);
    pMenuBar->addMenu(pHelp);
    pMenuBar->setProperty("cssClass", "mainWin");

    setMenuBar(pMenuBar);
}

void ZenoMainWindow::initDocks() {
    QWidget *p = takeCentralWidget();
    if (p)
        delete p;

    setDockNestingEnabled(true);

    m_viewDock = new ZenoDockWidget("view", this);
    m_viewDock->setObjectName(uniqueDockObjName(DOCK_VIEW));
    m_viewDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    DisplayWidget *view = new DisplayWidget(this);
    m_viewDock->setWidget(DOCK_VIEW, view);

    m_parameter = new ZenoDockWidget("parameter", this);
    m_parameter->setObjectName(uniqueDockObjName(DOCK_NODE_PARAMS));
    m_parameter->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_parameter->setWidget(DOCK_NODE_PARAMS, new ZenoPropPanel);

    m_editor = new ZenoDockWidget("", this);
    m_editor->setObjectName(uniqueDockObjName(DOCK_EDITOR));
    m_editor->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_pEditor = new ZenoGraphsEditor(this);
    m_editor->setWidget(DOCK_EDITOR, m_pEditor);
    // m_editor->setWidget(DOCK_EDITOR, new ZenoGraphsEditor(this));

    m_logger = new ZenoDockWidget("logger", this);
    m_logger->setObjectName(uniqueDockObjName(DOCK_LOG));
    m_logger->setObjectName(QString::fromUtf8("dock_logger"));
    m_logger->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_logger->setWidget(DOCK_LOG, new ZlogPanel);

    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *pDock : docks)
    {
        connect(pDock, SIGNAL(maximizeTriggered()), this, SLOT(onMaximumTriggered()));
        connect(pDock, SIGNAL(splitRequest(bool)), this, SLOT(onSplitDock(bool)));
    }
}

void ZenoMainWindow::onMaximumTriggered()
{
    ZenoDockWidget *pDockWidget = qobject_cast<ZenoDockWidget *>(sender());

    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *pDock : docks)
    {
        if (pDock != pDockWidget)
        {
            pDock->close();
        }
    }
}


void ZenoMainWindow::directlyRunRecord(const ZENO_RECORD_RUN_INITPARAM& param)
{
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
}

void ZenoMainWindow::updateViewport(const QString& action)
{
    //todo: temp code for single view.
    DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    if (view)
        view->updateFrame(action);
}
DisplayWidget* ZenoMainWindow::getDisplayWidget()
{
    DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    if (view)
        return view;
    return nullptr;
}

void ZenoMainWindow::onRunFinished()
{
    DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    if (view)
        view->onFinished();
}

void ZenoMainWindow::onSplitDock(bool bHorzontal)
{
    ZenoDockWidget *pDockWidget = qobject_cast<ZenoDockWidget *>(sender());
    ZenoDockWidget *pDock = new ZenoDockWidget("", this);
    pDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);

    if (ZenoGraphsEditor *pEditor = qobject_cast<ZenoGraphsEditor *>(pDockWidget->widget()))
    {
        ZenoGraphsEditor *pEditor2 = new ZenoGraphsEditor(this);
        pDock->setWidget(DOCK_EDITOR, pEditor2);
        //only one model.
        pEditor2->resetModel(zenoApp->graphsManagment()->currentModel());
        splitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
    }
    else
    {
        splitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
    }
    connect(pDock, SIGNAL(maximizeTriggered()), this, SLOT(onMaximumTriggered()));
    connect(pDock, SIGNAL(splitRequest(bool)), this, SLOT(onSplitDock(bool)));
}

void ZenoMainWindow::openFileDialog() {
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

void ZenoMainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);

    adjustDockSize();
}

void ZenoMainWindow::closeEvent(QCloseEvent *event) {
    this->saveQuit();
    // todo: event->ignore() when saveQuit returns false?
    QMainWindow::closeEvent(event);
}

void ZenoMainWindow::adjustDockSize() {
    //temp: different layout
    float height = size().height();
    int dockHeightA = 0.50 * height;
    int dockHeightB = 0.50 * height;

    QList<QDockWidget *> docks = {m_viewDock, m_editor};
    QList<int> dockSizes = {dockHeightA, dockHeightB};
    resizeDocks(docks, dockSizes, Qt::Vertical);
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
            ZlogPanel* pPanel = new ZlogPanel;
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
    DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    TIMELINE_INFO info;
    if (view)
    {
        info = view->timelineInfo();
    }
    return info;
}

void ZenoMainWindow::resetTimeline(TIMELINE_INFO info)
{
    DisplayWidget* view = qobject_cast<DisplayWidget*>(m_viewDock->widget());
    if (view)
    {
        view->resetTimeline(info);
    }
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
    auto docks = findChildren<ZenoDockWidget*>(QString(), Qt::FindDirectChildrenOnly);

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

void ZenoMainWindow::verticalLayout()
{
    addDockWidget(Qt::TopDockWidgetArea, m_viewDock);
    splitDockWidget(m_viewDock, m_editor, Qt::Vertical);
    splitDockWidget(m_editor, m_parameter, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_logger, Qt::Horizontal);
}

void ZenoMainWindow::onlyEditorLayout()
{
    verticalLayout();
    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *dock : docks)
    {
        if (dock->type() != DOCK_EDITOR)
        {
            dock->close();
        }
    }
}

void ZenoMainWindow::onNodesSelected(const QModelIndex &subgIdx, const QModelIndexList &nodes, bool select) {
    //dispatch to all property panel.
    auto docks = findChildren<ZenoDockWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget *dock : docks) {
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
void ZenoMainWindow::doFrameUpdate(int frame) {
    if(liveHttpServer->clients.empty())
        return;

    std::cout << "====== Frame " << frame << "\n";
    auto viewport = zenoApp->getMainWindow()->getDisplayWidget()->getViewportWidget();
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
