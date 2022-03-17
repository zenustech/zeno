#include <QApplication>
#include "style/zenostyle.h"
#include <comctrl/ziconbutton.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <comctrl/ztoolbutton.h>


int main(int argc, char *argv[]) 
{
	ZenoApplication a(argc, argv);
	a.setStyle(new ZenoStyle);

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
