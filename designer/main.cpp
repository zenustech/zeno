#include "designermainwin.h"
#include <style/zenostyle.h>

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	app.setStyle(new ZenoStyle);

	DesignerMainWin win;
	win.showMaximized();

	return app.exec();
}