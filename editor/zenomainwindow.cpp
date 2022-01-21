#include "zenomainwindow.h"
#include <comctrl/zenodockwidget.h>
#include "nodesview/znodeseditwidget.h"
#include "panel/zenodatapanel.h"
#include "timeline/ztimeline.h"
#include "tmpwidgets/ztoolbar.h"
#include "viewport/viewportwidget.h"
#include "viewport/zenovis.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <model/graphsmodel.h>
#include "launch/corelaunch.h"
#include "nodesview/zenographseditor.h"
#include <io/zsgreader.h>
#include <io/zsgwriter.h>


ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
{
    init();
}

void ZenoMainWindow::init()
{
    initMenu();
    initDocks();
    //houdiniStyleLayout();
    simplifyLayout();
    //readHoudiniStyleLayout();
}

void ZenoMainWindow::initMenu()
{
    QMenuBar *pMenuBar = new QMenuBar(this);
    if (!pMenuBar)
        return;

    pMenuBar->setMaximumHeight(26);//todo: sizehint

    QMenu *pFile = new QMenu(tr("File"));
    {
        QAction* pAction = new QAction(tr("New"), pFile);
        QMenu* pNewMenu = new QMenu;
        QAction* pNewGraph = pNewMenu->addAction("New Graph");

        pAction->setMenu(pNewMenu);

        pFile->addAction(pAction);

        pAction = new QAction(tr("Open"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+O")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(openFileDialog()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Save"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+S")));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Save As"), pFile);
        pAction->setCheckable(false);
        connect(pAction, SIGNAL(triggered()), this, SLOT(saveAs()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Import"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Shift+O")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(importGraph()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Export"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Shift+E")));
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

    QMenu *pRender = new QMenu(tr("Render"));

    QMenu *pView = new QMenu(tr("View"));

    QMenu *pWindow = new QMenu(tr("Window"));

    QMenu *pHelp = new QMenu(tr("Help"));

    pMenuBar->addMenu(pFile);
    pMenuBar->addMenu(pEdit);
    pMenuBar->addMenu(pRender);
    pMenuBar->addMenu(pView);
    pMenuBar->addMenu(pWindow);
    pMenuBar->addMenu(pHelp);

    setMenuBar(pMenuBar);
}

void ZenoMainWindow::initDocks()
{
    QWidget *p = takeCentralWidget();
    if (p)
        delete p;

    setDockNestingEnabled(true);

    //m_shapeBar = new ZenoDockWidget(this);
    //m_shapeBar->setObjectName(QString::fromUtf8("dock_shape"));
    //m_shapeBar->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    //m_shapeBar->setWidget(new ZShapeBar);

    //m_toolbar = new ZenoDockWidget(this);
    //m_toolbar->setObjectName(QString::fromUtf8("dock_toolbar"));
    //m_toolbar->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    //m_toolbar->setWidget(new ZToolbar);

    m_viewDock = new ZenoDockWidget("view", this);
    m_viewDock->setObjectName(QString::fromUtf8("dock_view"));
    m_viewDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    DisplayWidget* view = new DisplayWidget;
    m_viewDock->setWidget(view);

    //m_parameter = new ZenoDockWidget("parameter", this);
    //m_parameter->setObjectName(QString::fromUtf8("dock_parameter"));
    //m_parameter->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    //m_parameter->setWidget(new QWidget);

    ZenoGraphsEditor* pEditor = new ZenoGraphsEditor;
    //pEditor->setGeometry(500, 500, 1000, 1000);
    //ZNodesEditWidget* pOldEditor = new ZNodesEditWidget;

    //m_data = new ZenoDockWidget("data", this);
    //m_data->setObjectName(QString::fromUtf8("dock_data"));
    //m_data->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);

    m_editor = new ZenoDockWidget("", this);
    m_editor->setObjectName(QString::fromUtf8("dock_editor"));
    m_editor->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_editor->setWidget(pEditor);

    m_timelineDock = new ZenoDockWidget(this);
    m_timelineDock->setObjectName(QString::fromUtf8("dock_timeline"));
    m_timelineDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    ZTimeline* pTimeline = new ZTimeline;
    m_timelineDock->setWidget(pTimeline);

	connect(&Zenvis::GetInstance(), SIGNAL(frameUpdated(int)), pTimeline, SLOT(onTimelineUpdate(int)));
	connect(pTimeline, SIGNAL(playForward(bool)), &Zenvis::GetInstance(), SLOT(startPlay(bool)));
	connect(pTimeline, SIGNAL(sliderValueChanged(int)), &Zenvis::GetInstance(), SLOT(setCurrentFrameId(int)));
    connect(pTimeline, SIGNAL(run(int)), this, SLOT(onRunClicked(int)));
    connect(this, SIGNAL(modelInited()), pEditor, SLOT(onModelInited()));

    QTimer* pTimer = new QTimer;
    connect(pTimer, SIGNAL(timeout()), view, SLOT(updateFrame()));
    pTimer->start(16);
}

void ZenoMainWindow::openFileDialog()
{
    GraphsManagment* pGraphs = zenoApp->graphsManagment();
    saveQuit();

    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;
    //todo: path validation
    GraphsModel* pModel = zenoApp->graphsManagment()->openZsgFile(filePath);
    pModel->initDescriptors();
    emit modelInited();
}

void ZenoMainWindow::saveQuit()
{
    GraphsManagment* pGraphs = zenoApp->graphsManagment();
    if (pGraphs->saveCurrent())
    {
        saveAs();
    }
    pGraphs->clear();
}

void ZenoMainWindow::saveAs()
{
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zensim Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty())
    {
        QString strContent = ZsgWriter::getInstance().dumpProgramStr(zenoApp->graphsManagment()->currentModel());
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) {
            qWarning() << Q_FUNC_INFO << "Failed to open" << path << f.errorString();
            return;
        }
        f.write(strContent.toUtf8());
        f.close();
    }
}

QString ZenoMainWindow::getOpenFileByDialog()
{
    const QString& initialPath = ".";
    QFileDialog fileDialog(this, tr("Open"), initialPath, "Zensim Graph File (*.zsg)\nAll Files (*)");
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setFileMode(QFileDialog::ExistingFile);
    fileDialog.setDirectory(initialPath);
    if (fileDialog.exec() != QDialog::Accepted)
        return "";

    QString filePath = fileDialog.selectedFiles().first();
    return filePath;
}

void ZenoMainWindow::houdiniStyleLayout()
{
    addDockWidget(Qt::TopDockWidgetArea, m_shapeBar);
    splitDockWidget(m_shapeBar, m_toolbar, Qt::Vertical);
    splitDockWidget(m_toolbar, m_timelineDock, Qt::Vertical);

    splitDockWidget(m_toolbar, m_viewDock, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_editor, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_data, Qt::Vertical);
    splitDockWidget(m_data, m_parameter, Qt::Horizontal);

    writeHoudiniStyleLayout();
}

void ZenoMainWindow::simplifyLayout()
{
    addDockWidget(Qt::TopDockWidgetArea, m_viewDock);
    splitDockWidget(m_viewDock, m_editor, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_timelineDock, Qt::Vertical);
}

void ZenoMainWindow::arrangeDocks2()
{
    addDockWidget(Qt::LeftDockWidgetArea, m_toolbar);
    splitDockWidget(m_toolbar, m_viewDock, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_parameter, Qt::Horizontal);

    splitDockWidget(m_viewDock, m_timelineDock, Qt::Vertical);
    splitDockWidget(m_parameter, m_data, Qt::Vertical);
    splitDockWidget(m_data, m_editor, Qt::Vertical);
    writeSettings2();
}

void ZenoMainWindow::arrangeDocks3()
{
    addDockWidget(Qt::TopDockWidgetArea, m_parameter);
    addDockWidget(Qt::LeftDockWidgetArea, m_toolbar);
    addDockWidget(Qt::BottomDockWidgetArea, m_timelineDock);
    splitDockWidget(m_toolbar, m_viewDock, Qt::Horizontal);
    splitDockWidget(m_viewDock, m_data, Qt::Vertical);
    splitDockWidget(m_data, m_editor, Qt::Vertical);
    writeHoudiniStyleLayout();
}

void ZenoMainWindow::writeHoudiniStyleLayout()
{
    QSettings settings("Zeno Inc.", "zeno2 ui1");
    settings.beginGroup("mainWindow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    settings.endGroup();
}

void ZenoMainWindow::writeSettings2()
{
    QSettings settings("Zeno Inc.", "zeno2 ui2");
    settings.beginGroup("mainWindow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    settings.endGroup();
}

void ZenoMainWindow::readHoudiniStyleLayout()
{
    QSettings settings("Zeno Inc.", "zeno2 ui1");
    settings.beginGroup("mainWindow");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("state").toByteArray());
    settings.endGroup();
}

void ZenoMainWindow::readSettings2()
{
    QSettings settings("Zeno Inc.", "zeno2 ui2");
    settings.beginGroup("mainWindow");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("state").toByteArray());
    settings.endGroup();
}

void ZenoMainWindow::onRunClicked(int nFrames)
{
    GraphsManagment* pGraphsMgr = zenoApp->graphsManagment();
    GraphsModel* pModel = pGraphsMgr->currentModel();
    launchProgram(pModel, nFrames);
}