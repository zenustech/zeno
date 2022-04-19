#include "zenomainwindow.h"
#include "dock/zenodockwidget.h"
#include "panel/zenodatapanel.h"
#include "panel/zenoproppanel.h"
#include "timeline/ztimeline.h"
#include "tmpwidgets/ztoolbar.h"
#include "viewport/viewportwidget.h"
#include "viewport/zenovis.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include "launch/corelaunch.h"
#include "nodesview/zenographseditor.h"
#include <zenoio/reader/zsgreader.h>
#include <zenoio/writer/zsgwriter.h>
#include <zeno/utils/log.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/util/uihelper.h>


ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , m_pEditor(nullptr)
    , m_viewDock(nullptr)
    , m_timelineDock(nullptr)
    , m_bInDlgEventloop(false)
{
    init();
    setContextMenuPolicy(Qt::NoContextMenu);

    setWindowTitle("Zeno Editor (github.com/zenustech/zeno)");
#ifdef __linux__
    if (char *p = std::getenv("ZENO_OPEN")) {
        printf("ZENO_OPEN: %s\n", p);
        openFile(p);
    }
#endif
}

ZenoMainWindow::~ZenoMainWindow()
{
}

void ZenoMainWindow::init()
{
    initMenu();
    initDocks();
    verticalLayout();
    //onlyEditorLayout();

    QPalette pal = palette();
	pal.setColor(QPalette::Window, QColor(11, 11, 11));
	setAutoFillBackground(true);
	setPalette(pal);
}

void ZenoMainWindow::initMenu()
{
    QMenuBar *pMenuBar = new QMenuBar(this);
    if (!pMenuBar)
        return;

    QMenu *pFile = new QMenu(tr("File"));
    {
        QAction* pAction = new QAction(tr("New"), pFile);
        QMenu* pNewMenu = new QMenu;
        QAction* pNewGraph = pNewMenu->addAction("New Scene");
        connect(pNewGraph, SIGNAL(triggered()), this, SLOT(onNewFile()));

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
        connect(pAction, SIGNAL(triggered()), this, SLOT(save()));
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

    QMenu* pDisplay = new QMenu(tr("Display"));
    {
        QAction* pAction = new QAction(tr("Show Grid"), this);
        pAction->setCheckable(true);
        pAction->setChecked(true);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().setShowGrid(pAction->isChecked());
        });

        pAction = new QAction(tr("Background Color"), this);
        pDisplay->addAction(pAction);

        pDisplay->addSeparator();

        pAction = new QAction(tr("Smooth Shading"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);

        pAction = new QAction(tr("Normal Check"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().setNormalCheck(pAction->isChecked());
        });

        pAction = new QAction(tr("Wireframe"), this);
        pAction->setCheckable(true);
        pAction->setChecked(false);
        pDisplay->addAction(pAction);
        connect(pAction, &QAction::triggered, this, [=]() {
            Zenovis::GetInstance().setRenderWireframe(pAction->isChecked());
        });

        pDisplay->addSeparator();

        pAction = new QAction(tr("Camera Keyframe"), this);
        pDisplay->addAction(pAction);

        pDisplay->addSeparator();

        pAction = new QAction(tr("Use English"), this);
        pAction->setCheckable(true);
        pAction->setChecked(true);
        pDisplay->addAction(pAction);
    }

    QMenu* pRecord = new QMenu(tr("Record"));
    {
        QAction* pAction = new QAction(tr("Screenshot"), this);
        pAction->setShortcut(QKeySequence("F12"));
        pRecord->addAction(pAction);

        pAction = new QAction(tr("Record Video"), this);
        pAction->setShortcut(QKeySequence(tr("Shift+F12")));
        pRecord->addAction(pAction);
    }

    QMenu *pRender = new QMenu(tr("Render"));

    QMenu *pView = new QMenu(tr("View"));
    {
        QAction* pAction = new QAction(tr("view"));
        pAction->setCheckable(true);
        pAction->setChecked(true);
        connect(pAction, &QAction::triggered, this, [=]() {
            onToggleDockWidget(DOCK_VIEW, pAction->isChecked());
        });
        pView->addAction(pAction);

        pAction = new QAction(tr("editor"));
        pAction->setCheckable(true);
        pAction->setChecked(true);
        connect(pAction, &QAction::triggered, this, [=]() {
            onToggleDockWidget(DOCK_EDITOR, pAction->isChecked());
            });
        pView->addAction(pAction);
    }

    QMenu *pWindow = new QMenu(tr("Window"));

    QMenu *pHelp = new QMenu(tr("Help"));

    pMenuBar->addMenu(pFile);
    pMenuBar->addMenu(pEdit);
    pMenuBar->addMenu(pRender);
    pMenuBar->addMenu(pView);
    pMenuBar->addMenu(pDisplay);
    pMenuBar->addMenu(pRecord);
    pMenuBar->addMenu(pWindow);
    pMenuBar->addMenu(pHelp);

    /* up-right-bottom-left */
    const QString& qssTemplate =
    "\
    QMenuBar {\
        background-color: rgb(28, 28, 28);\
        spacing: %1px; \
        color: rgba(255,255,255,0.50);\
    }\
    \
    QMenuBar::item {\
        padding: %2px %3px %4px %5px;\
        background: transparent;\
    }\
    \
    QMenuBar::item:selected {\
        background: #4B9EF4;\
    }\
    \
    QMenuBar::item:pressed {\
        background: #4B9EF4;\
    }\
    ";

    qreal spacing = ZenoStyle::dpiScaled(3), padding_left = ZenoStyle::dpiScaled(8),
        padding_right = ZenoStyle::dpiScaled(8), padding_top = ZenoStyle::dpiScaled(4), padding_bottom = ZenoStyle::dpiScaled(4);
    QString qss = qssTemplate.arg(spacing).arg(padding_top).arg(padding_right).arg(padding_bottom).arg(padding_left);
    pMenuBar->setStyleSheet(qss);

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
	DisplayWidget* view = new DisplayWidget(this);
	m_viewDock->setWidget(DOCK_VIEW, view);
	m_docks.insert(DOCK_VIEW, m_viewDock);

    m_parameter = new ZenoDockWidget("parameter", this);
    m_parameter->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_parameter->setWidget(DOCK_NODE_PARAMS, new ZenoPropPanel);
    //m_parameter->setWidget(new QWidget);

    //m_pEditor = new ZenoGraphsEditor;
    //pEditor->setGeometry(500, 500, 1000, 1000);
    //ZNodesEditWidget* pOldEditor = new ZNodesEditWidget;

    //m_data = new ZenoDockWidget("data", this);
    //m_data->setObjectName(QString::fromUtf8("dock_data"));
    //m_data->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);

    m_editor = new ZenoDockWidget("", this);
    m_editor->setObjectName(QString::fromUtf8("dock_editor"));
    m_editor->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_pEditor = new ZenoGraphsEditor(this);
    m_editor->setWidget(DOCK_EDITOR, m_pEditor);
    m_docks.insert(DOCK_EDITOR, m_editor);

    for (QMap<DOCK_TYPE, ZenoDockWidget*>::iterator it = m_docks.begin(); it != m_docks.end(); it++)
    {
        ZenoDockWidget* pDock = it.value();
        connect(pDock, SIGNAL(maximizeTriggered()), this, SLOT(onMaximumTriggered()));
        connect(pDock, SIGNAL(splitRequest(bool)), this, SLOT(onSplitDock(bool)));
    }
}

void ZenoMainWindow::onMaximumTriggered()
{
    ZenoDockWidget* pDockWidget = qobject_cast<ZenoDockWidget*>(sender());
    for (QMap<DOCK_TYPE, ZenoDockWidget*>::iterator it = m_docks.begin(); it != m_docks.end(); it++)
    {
        ZenoDockWidget* pDock = it.value();
        if (pDock != pDockWidget)
        {
            pDock->close();
        }
    }
}

void ZenoMainWindow::onSplitDock(bool bHorzontal)
{
    ZenoDockWidget* pDockWidget = qobject_cast<ZenoDockWidget*>(sender());
    ZenoDockWidget* pDock = new ZenoDockWidget("", this);
    pDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);

    if (ZenoGraphsEditor* pEditor = qobject_cast<ZenoGraphsEditor*>(pDockWidget->widget()))
    {
        ZenoGraphsEditor* pEditor2 = new ZenoGraphsEditor(this);
        pDock->setWidget(DOCK_EDITOR, pEditor2);
        //only one model.
        pEditor2->resetModel(zenoApp->graphsManagment()->currentModel());
        m_docks.insert(DOCK_EDITOR, pDock);
        splitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
    }
    else
    {
        splitDockWidget(pDockWidget, pDock, bHorzontal ? Qt::Horizontal : Qt::Vertical);
    }
	connect(pDock, SIGNAL(maximizeTriggered()), this, SLOT(onMaximumTriggered()));
	connect(pDock, SIGNAL(splitRequest(bool)), this, SLOT(onSplitDock(bool)));
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

void ZenoMainWindow::onNewFile()
{
    saveQuit();
    zenoApp->graphsManagment()->newFile();
}

void ZenoMainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);

    adjustDockSize();
}

void ZenoMainWindow::adjustDockSize()
{
    //temp: different layout
	float height = size().height();
	int dockHeightA = 0.50 * height;
	int dockHeightB = 0.50 * height;

	QList<QDockWidget*> docks = { m_viewDock, m_editor };
	QList<int> dockSizes = { dockHeightA, dockHeightB };
	resizeDocks(docks, dockSizes, Qt::Vertical);
}

void ZenoMainWindow::importGraph()
{
	QString filePath = getOpenFileByDialog();
	if (filePath.isEmpty())
		return;

	//todo: path validation
    auto pGraphs = zenoApp->graphsManagment();
    pGraphs->importGraph(filePath);
}

bool ZenoMainWindow::openFile(QString filePath)
{
    auto pGraphs = zenoApp->graphsManagment();
    IGraphsModel* pModel = pGraphs->openZsgFile(filePath);
    if (!pModel)
        return false;
    return true;
}

void ZenoMainWindow::onToggleDockWidget(DOCK_TYPE type, bool bShow)
{
    QList<ZenoDockWidget*> list = m_docks.values(type);
    for (auto dock : list)
    {
        if (bShow)
            dock->show();
        else
            dock->close();
    }
}

void ZenoMainWindow::onDockSwitched(DOCK_TYPE type)
{
    ZenoDockWidget* pDock = qobject_cast<ZenoDockWidget*>(sender());
    switch (type)
    {
        case DOCK_EDITOR:
        {
            ZenoGraphsEditor* pEditor2 = new ZenoGraphsEditor(this);
            pEditor2->resetModel(zenoApp->graphsManagment()->currentModel());
            pDock->setWidget(type, pEditor2);
            break;
        }
        case DOCK_VIEW:
        {
            //complicated opengl framework.
            //DisplayWidget* view = new DisplayWidget;
            //pDock->setWidget(view);
            break;
        }
        case DOCK_NODE_PARAMS:
        {
            ZenoPropPanel* pWidget = new ZenoPropPanel;
            pDock->setWidget(type, pWidget);
            break;
        }
        case DOCK_NODE_DATA:
        {
			QWidget* pWidget = new QWidget;
			QPalette pal = pWidget->palette();
			pal.setColor(QPalette::Window, QColor(0, 0, 255));
			pWidget->setAutoFillBackground(true);
			pWidget->setPalette(pal);
			pDock->setWidget(type, pWidget);
            break;
        }
    }
}

void ZenoMainWindow::saveQuit()
{
    auto pGraphsMgm = zenoApp->graphsManagment();
    Q_ASSERT(pGraphsMgm);
    IGraphsModel* pModel = pGraphsMgm->currentModel();
    if (pModel && pModel->isDirty())
    {
        QMessageBox msgBox = QMessageBox(QMessageBox::Question, "Save", "Save changes?", QMessageBox::Yes | QMessageBox::No, this);
        QPalette pal = msgBox.palette();
        pal.setBrush(QPalette::WindowText, QColor(0, 0, 0));
        msgBox.setPalette(pal);
        int ret = msgBox.exec();
		if (ret & QMessageBox::Yes)
		{
			saveAs();
		}
    }
    pGraphsMgm->clear();
}

void ZenoMainWindow::save()
{
	auto pGraphsMgm = zenoApp->graphsManagment();
	Q_ASSERT(pGraphsMgm);
	IGraphsModel* pModel = pGraphsMgm->currentModel();
    if (pModel)
    {
        QString currFilePath = pModel->filePath();
		if (currFilePath.isEmpty())
			return saveAs();
		saveFile(currFilePath);
    }
}

bool ZenoMainWindow::saveFile(QString filePath)
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QString strContent = ZsgWriter::getInstance().dumpProgramStr(pModel);
    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << f.errorString();
        zeno::log_error("Failed to open zsg for write: {} ({})",
                       filePath.toStdString(), f.errorString().toStdString());
        return false;
    }
    f.write(strContent.toUtf8());
    f.close();
    pModel->setFilePath(filePath);
    pModel->clearDirty();
    return true;
}

bool ZenoMainWindow::inDlgEventLoop() const
{
    return m_bInDlgEventloop;
}

void ZenoMainWindow::saveAs()
{
    VarToggleScope scope(&m_bInDlgEventloop);
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zensim Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty())
    {
        saveFile(path);
    }
}

QString ZenoMainWindow::getOpenFileByDialog()
{
    VarToggleScope scope(&m_bInDlgEventloop);
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

void ZenoMainWindow::verticalLayout()
{
    addDockWidget(Qt::TopDockWidgetArea, m_viewDock);
    splitDockWidget(m_viewDock, m_editor, Qt::Vertical);
    splitDockWidget(m_editor, m_parameter, Qt::Horizontal);
}

void ZenoMainWindow::onlyEditorLayout()
{
    verticalLayout();
	for (QMap<DOCK_TYPE, ZenoDockWidget*>::iterator it = m_docks.begin(); it != m_docks.end(); it++)
	{
		ZenoDockWidget* pDock = it.value();
		if (pDock != m_editor)
		{
			pDock->close();
		}
	}
}

void ZenoMainWindow::simpleLayout2()
{
    addDockWidget(Qt::TopDockWidgetArea, m_viewDock);
    splitDockWidget(m_viewDock, m_timelineDock, Qt::Vertical);
    splitDockWidget(m_timelineDock, m_editor, Qt::Vertical);
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

void ZenoMainWindow::onRunClicked(int beginFrame, int endFrame)
{
    auto pGraphsMgr = zenoApp->graphsManagment();
    IGraphsModel* pModel = pGraphsMgr->currentModel();
    GraphsModel* pLegacy = qobject_cast<GraphsModel*>(pModel);
    launchProgram(pLegacy, beginFrame, endFrame);
}

void ZenoMainWindow::onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    //dispatch to all property panel.
    auto docks = findChildren<ZenoDockWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (ZenoDockWidget* dock : docks)
    {
        dock->onNodesSelected(subgIdx, nodes, select);
    }
}