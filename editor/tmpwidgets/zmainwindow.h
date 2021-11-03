#ifndef __ZMAINWINDOW_H__
#define __ZMAINWINDOW_H__


#include <kddockwidgets/DockWidget.h>
#include <kddockwidgets/MainWindow.h>

class ZTimeline;
class ZMenuBar;

class ZMainWindow : public KDDockWidgets::MainWindow
{
	Q_OBJECT
public:
	ZMainWindow(QWidget* parent = nullptr);
	~ZMainWindow();

private:
	void initMenu();

	KDDockWidgets::DockWidget* m_viewportDockWidget;
	KDDockWidgets::DockWidget* m_nodesDockWidget;
	KDDockWidgets::DockWidget* m_properDockWidget;

	ZMenuBar* m_menu;
	ZTimeline* m_timeline;
};

#endif