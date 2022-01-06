#include <QApplication>
#include "style/zenostyle.h"
#include <comctrl/ziconbutton.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"


int main(int argc, char *argv[]) 
{
	ZenoApplication a(argc, argv);
	a.setStyle(new ZenoStyle);

	QPalette palette = a.palette();
	palette.setColor(QPalette::Window, QColor(11, 11, 11));
	palette.setColor(QPalette::WindowText, Qt::white);
	a.setPalette(palette);

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
