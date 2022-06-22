#include <QApplication>
#include <zeno/extra/assetDir.h>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"


int main(int argc, char *argv[]) 
{
    startUp();
#ifdef ZENO_MULTIPROCESS
    if (argc == 3 && !strcmp(argv[1], "-runner")) {
        extern int runner_main(int sessionid);
        return runner_main(atoi(argv[2]));
    }
#endif
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    zeno::setExecutableDir(a.applicationDirPath().toStdString());

    QTranslator t;
    QSettings settings("ZenusTech", "Zeno");
    QVariant use_chinese = settings.value("use_chinese");

    if (use_chinese.isNull() || use_chinese.toBool()) {
        if (t.load(":languages/zh.qm")) {
            a.installTranslator(&t);
        }
    }

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
