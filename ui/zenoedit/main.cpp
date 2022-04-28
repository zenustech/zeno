#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"


int main(int argc, char *argv[]) 
{
#ifdef ZENO_MULTIPROCESS
    if (argc == 2 && !strcmp(argv[1], "-runner")) {
        extern int runner_main();
        return runner_main();
    }
#endif

	ZenoApplication a(argc, argv);
	a.setStyle(new ZenoStyle);

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
