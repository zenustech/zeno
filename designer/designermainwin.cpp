#include "designermainwin.h"
#include "layerwidget.h"
#include "viewmdiarea.h"
#include "propertypane.h"
#include "styletabwidget.h"
#include "ztabpanel.h"


DesignerMainWin::DesignerMainWin()
    : m_pMenubar(nullptr)
    , m_pLayerWidget(nullptr)
    , m_pMdiArea(nullptr)
    , m_properties(nullptr)
    , m_tabWidget(nullptr)
{
    initMenu();
    initWidgets();
	initConnections();
	resetModels();
}

void DesignerMainWin::initWidgets()
{
    QHBoxLayout* pLayout = new QHBoxLayout;

    QSplitter* pSplitter = new QSplitter;
    pSplitter->setOrientation(Qt::Horizontal);
    pSplitter->setFrameShape(QFrame::NoFrame);
    pSplitter->setHandleWidth(3);

    m_pLayerWidget = new LayerWidget;
    pSplitter->addWidget(m_pLayerWidget);

    m_tabWidget = new StyleTabWidget(m_pLayerWidget);
	pSplitter->addWidget(m_tabWidget);

    m_properties = new ZTabPanel;
    pSplitter->addWidget(m_properties);

    pLayout->addWidget(pSplitter);

    QWidget* centralWidget = new QWidget;
    centralWidget->setLayout(pLayout);

    setCentralWidget(centralWidget);
}

void DesignerMainWin::initConnections()
{
	connect(m_tabWidget, SIGNAL(tabviewActivated(QStandardItemModel*)), m_pLayerWidget, SLOT(setModel(QStandardItemModel*)));
}

void DesignerMainWin::resetModels()
{
	m_pLayerWidget->setModel(m_tabWidget->getCurrentModel(), m_tabWidget->getSelectionModel());
}

void DesignerMainWin::initMdiWindows()
{
    QMdiSubWindow* mdiWin = new QMdiSubWindow;
    mdiWin->setWindowTitle(tr("node"));

    QMdiSubWindow* mdiWin2 = new QMdiSubWindow;
    mdiWin2->setWindowTitle(tr("mdiWin2"));

    m_pMdiArea->addSubWindow(mdiWin);
    m_pMdiArea->addSubWindow(mdiWin2);

    m_pMdiArea->setViewMode(QMdiArea::TabbedView);
    m_pMdiArea->show();
}

void DesignerMainWin::initMenu()
{
    m_pMenubar = new QMenuBar;
    QMenu* pFile = new QMenu(tr("File"));
	{
		QAction* pAction = new QAction(tr("New"), pFile);
		pAction->setCheckable(false);
		pFile->addAction(pAction);

		pAction = new QAction(tr("Open"), pFile);
		pAction->setCheckable(false);
		pFile->addAction(pAction);

		pAction = new QAction(tr("Save"), pFile);
		pAction->setCheckable(false);
		pFile->addAction(pAction);

		pAction = new QAction(tr("Quit"), pFile);
		pAction->setCheckable(false);
		pFile->addAction(pAction);
	}

	QMenu* pEdit = new QMenu(tr("Edit"));
	{
		QAction* pAction = new QAction(tr("Undo"), pEdit);
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

	QMenu* pLayout = new QMenu(tr("Layout"));

	QMenu* pPlay = new QMenu(tr("Play"));

	QMenu* pView= new QMenu(tr("View"));

	QMenu* pHelp = new QMenu(tr("Help"));

	m_pMenubar->addMenu(pFile);
	m_pMenubar->addMenu(pEdit);
	m_pMenubar->addMenu(pLayout);
	m_pMenubar->addMenu(pPlay);
	m_pMenubar->addMenu(pView);
	m_pMenubar->addMenu(pHelp);

    setMenuBar(m_pMenubar);
}