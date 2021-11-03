#include "zmainwindow.h"
#include <kddockwidgets/Config.h>
#include "ztoolbar.h"
#include "../timeline/ztimeline.h"


ZMainWindow::ZMainWindow(QWidget* parent)
	: KDDockWidgets::MainWindow(QStringLiteral("Zeno"), KDDockWidgets::MainWindowOption_None, parent)
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

	auto toolbarDock = new KDDockWidgets::DockWidget(QStringLiteral("toolbar"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions());
	auto tabWidget = new QTabWidget;
	tabWidget->addTab(new ZShapeBar, "Create");
	tabWidget->addTab(new ZTextureBar, "Texture");
	tabWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	toolbarDock->setWidget(tabWidget);
	addDockWidget(toolbarDock, KDDockWidgets::Location_OnTop);

	m_properDockWidget = new KDDockWidgets::DockWidget(QStringLiteral("properties"));
	auto propertiesWidget = new QTabWidget;
	propertiesWidget->addTab(new QWidget, "Properties");
	propertiesWidget->addTab(new QWidget, "Shapes");
	m_properDockWidget->setWidget(propertiesWidget);

	auto timelineDock = new KDDockWidgets::DockWidget(QStringLiteral("timeline"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions());
	m_timeline = new ZTimeline;
	m_timeline->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	timelineDock->setWidget(m_timeline);
	addDockWidget(timelineDock, KDDockWidgets::Location_OnBottom);

	addDockWidget(m_properDockWidget, KDDockWidgets::Location_OnRight);

	auto minitoolbar = new KDDockWidgets::DockWidget(QStringLiteral("minitoolbar"),
		KDDockWidgets::DockWidgetBase::Options(),
		KDDockWidgets::DockWidgetBase::LayoutSaverOptions());
	auto toolbar = new ZToolbar;
	toolbar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
	minitoolbar->setWidget(toolbar);
	addDockWidget(minitoolbar, KDDockWidgets::Location_OnLeft);

	setWindowIcon(QIcon(":/icons/zenus.png"));
}

ZMainWindow::~ZMainWindow()
{
}

void ZMainWindow::initMenu()
{
}