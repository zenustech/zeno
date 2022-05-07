#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"


int main(int argc, char *argv[]) 
{
#ifdef ZENO_MULTIPROCESS
    if (argc == 3 && !strcmp(argv[1], "-runner")) {
        extern int runner_main(int sessionid);
        return runner_main(atoi(argv[2]));
    }
#endif

    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
