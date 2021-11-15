#include "designermainwin.h"

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	DesignerMainWin win;
	win.showMaximized();

	return app.exec();
}