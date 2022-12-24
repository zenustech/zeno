#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"


int main(int argc, char *argv[]) 
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    startUp();

#ifdef ZENO_MULTIPROCESS
    if (argc >= 3 && !strcmp(argv[1], "-runner")) {
        extern int runner_main(int sessionid, int port, char* cachedir);
        int sessionid = atoi(argv[2]);
        int port = -1;
        char* cachedir = nullptr;
        if (argc >= 5 && !strcmp(argv[3], "-port"))
            port = atoi(argv[4]);
        if (argc >= 7 && !strcmp(argv[5], "-cachedir"))
            cachedir = argv[6];
        return runner_main(sessionid, port, cachedir);
    }
#endif

    if (argc >= 3 && !strcmp(argv[1], "-invoke")) {
        extern int invoke_main(int argc, char *argv[]);
        return invoke_main(argc - 2, argv + 2);
    }

    if (argc >= 3 && !strcmp(argv[1], "-offline")) {
        extern int offline_main(const char *zsgfile, int beginFrame, int endFrame);
        int begin = 0, end = 0;
        if (argc >= 5 && !strcmp(argv[3], "-begin"))
            begin = atoi(argv[4]);
        if (argc >= 5 && !strcmp(argv[3], "-end"))
            end = atoi(argv[4]);
        if (argc >= 7 && !strcmp(argv[5], "-begin"))
            begin = atoi(argv[6]);
        if (argc >= 7 && !strcmp(argv[5], "-end"))
            end = atoi(argv[6]);
        return offline_main(argv[2], begin, end);
    }

    //entrance for the zenoedit-player. todo: cmdParser
    if (0)
    {
        ZenoMainWindow tempWindow;
        tempWindow.showMaximized();

        ZENO_RECORD_RUN_INITPARAM initParam;
        initParam.sZsgPath = "E:\\zeno-fixbug\\once-bug.zsg";
        initParam.sPath = "E:\\zeno-fixbug\\recordpath";
        initParam.iFps = 24;
        initParam.iBitrate = 200000;
        initParam.iSFrame = 0;
        initParam.iFrame = 10;
        initParam.sPixel = "1200x800";

        tempWindow.directlyRunRecord(initParam);
        return a.exec();
    }

    QTranslator t;
    QSettings settings(zsCompanyName, zsEditor);
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
