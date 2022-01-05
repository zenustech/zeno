#include "zmainwindow.h"
#include <kddockwidgets/Config.h>
#include "ztoolbar.h"
#include "../timeline/ztimeline.h"
#include "dmmenu.h"
#include <kddockwidgets/Config.h>
#include "../nodesview/znodeswebview.h"
#include "../nodesview/znodeseditwidget.h"


class fakeViewportWidget : public QWidget
{
public:
	fakeViewportWidget(QWidget* parent = nullptr) : QWidget(parent) {
		QVBoxLayout* pLayout = new QVBoxLayout;
		pLayout->setContentsMargins(0, 0, 0, 0);

		QMenuBar* menuBar = new QMenuBar;
		menuBar->setMaximumHeight(26);

		QDMDisplayMenu* menuDisplay = new QDMDisplayMenu;
		menuBar->addMenu(menuDisplay);
		QDMRecordMenu* recordDisplay = new QDMRecordMenu;
		menuBar->addMenu(recordDisplay);

		pLayout->addWidget(menuBar);

		QHBoxLayout* pHBoxLayout = new QHBoxLayout;
		pHBoxLayout->addStretch();
		QLabel* pTipLabel = new QLabel("The central widget can be a viewport");
		QPalette palette = pTipLabel->palette();
		palette.setColor(pTipLabel->foregroundRole(), Qt::white);
		pTipLabel->setPalette(palette);
		pHBoxLayout->addWidget(pTipLabel);
		pHBoxLayout->addStretch();

		pLayout->addLayout(pHBoxLayout);

		setLayout(pLayout);
		
		setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
	}

	QSize sizeHint() const override {
		return QSize(width(), 200);
	}
};


ZMainWindow::ZMainWindow(QWidget* parent)
	: KDDockWidgets::MainWindow(QStringLiteral("Zeno"), KDDockWidgets::MainWindowOption_HasCentralWidgetAndMenubar, parent)
	, m_viewportDockWidget(nullptr)
	, m_nodesDockWidget(nullptr)
	, m_properDockWidget(nullptr)
	, m_menu(nullptr)
	, m_timeline(nullptr)
{
	QPalette palette;
	palette.setBrush(QPalette::Window, QColor(0, 0, 0));
	setPalette(palette);

	setWindowTitle("Zeno");
	initMenu();

	addToCentralFrame(new fakeViewportWidget);

	auto toolbarDock = new KDDockWidgets::DockWidget(QStringLiteral("toolbar"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions(),
		KDDockWidgets::DockWidgetBase::TitleBarStyle::TitleStyle_ToolBarVertical);
	auto tabWidget = new QTabWidget;
	tabWidget->addTab(new ZShapeBar, "Create");
	tabWidget->addTab(new ZTextureBar, "Texture");
	//tabWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	toolbarDock->setWidget(tabWidget);
	addDockWidget(toolbarDock, KDDockWidgets::Location_OnTop);

	m_properDockWidget = new KDDockWidgets::DockWidget(QStringLiteral("properties"));
	auto propertiesWidget = new QTabWidget;
	propertiesWidget->addTab(new QWidget, "Properties");
	propertiesWidget->addTab(new QWidget, "Shapes");
	m_properDockWidget->setWidget(propertiesWidget);

	addDockWidget(m_properDockWidget, KDDockWidgets::Location_OnRight);

	auto minitoolbar = new KDDockWidgets::DockWidget(QStringLiteral("minitoolbar"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions(),
		KDDockWidgets::DockWidgetBase::TitleBarStyle::TitleStyle_ToolBarHorizontal);
	auto toolbar = new ZToolbar;
	//toolbar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
	minitoolbar->setWidget(toolbar);
	addDockWidget(minitoolbar, KDDockWidgets::Location_OnLeft);

	auto timelineDock = new KDDockWidgets::DockWidget(QStringLiteral("timeline"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions(),
		KDDockWidgets::DockWidgetBase::TitleBarStyle::TitleStyle_ToolBarVertical);
	m_timeline = new ZTimeline;
	//m_timeline->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	timelineDock->setWidget(m_timeline);
	addDockWidget(timelineDock, KDDockWidgets::Location_OnBottom);

	m_nodesDockWidget = new KDDockWidgets::DockWidget(QStringLiteral("nodes"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions(),		
		KDDockWidgets::DockWidgetBase::TitleBarStyle::TitleStyle_ToolBarHorizontal);
	auto nodesView = new ZNodesEditWidget;
	m_nodesDockWidget->setWidget(nodesView);
	addDockWidget(m_nodesDockWidget, KDDockWidgets::Location_OnBottom);

	setWindowIcon(QIcon(":/icons/zenus.png"));
}

ZMainWindow::~ZMainWindow()
{
}

void ZMainWindow::initMenu()
{
	QMenuBar* pMenu = this->MenuBar();
	if (!pMenu)
		return;

	pMenu->setMaximumHeight(26); //todo: sizehint

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

	QMenu* pRender = new QMenu(tr("Render"));

	QMenu* pView = new QMenu(tr("View"));

	QMenu* pWindow = new QMenu(tr("Window"));

	QMenu* pHelp = new QMenu(tr("Help"));

	pMenu->addMenu(pFile);
	pMenu->addMenu(pEdit);
	pMenu->addMenu(pRender);
	pMenu->addMenu(pView);
	pMenu->addMenu(pWindow);
	pMenu->addMenu(pHelp);
}