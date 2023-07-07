#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"

/* debug cutsom layout: ZGraphicsLayout */
//#define DEBUG_ZENOGV_LAYOUT
//#define DEBUG_NORMAL_WIDGET

#ifdef DEBUG_ZENOGV_LAYOUT
#include <zenoui/comctrl/gv/gvtestwidget.h>
#endif

#ifdef DEBUG_NORMAL_WIDGET
#include <zenoui/comctrl/testwidget.h>
#endif


int main(int argc, char *argv[]) 
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

#ifdef DEBUG_NORMAL_WIDGET
    TestNormalWidget wid;
    wid.show();
    return a.exec();
#endif

#ifdef DEBUG_ZENOGV_LAYOUT
    TestGraphicsView view;
    view.show();
    return a.exec();
#endif

    startUp();

#ifdef ZENO_MULTIPROCESS
    if (argc >= 3 && !strcmp(argv[1], "-runner")) {
        extern int runner_main(int sessionid, int port, char *cachedir, bool cacheLightCameraOnly, bool cacheMaterialOnly);
        int sessionid = atoi(argv[2]);
        int port = -1;
        char* cachedir = nullptr;
        bool cacheLightCameraOnly = false;
        bool cacheMaterialOnly = false;
        if (argc >= 5 && !strcmp(argv[3], "-port"))
            port = atoi(argv[4]);
        if (argc >= 7 && !strcmp(argv[5], "-cachedir"))
            cachedir = argv[6];
        if (argc >= 9 && !strcmp(argv[7], "-cacheLightCameraOnly"))
            cacheLightCameraOnly = atoi(argv[8]);
        if (argc >= 11 && !strcmp(argv[9], "-cacheMaterialOnly"))
            cacheMaterialOnly = atoi(argv[10]);
        return runner_main(sessionid, port, cachedir, cacheLightCameraOnly, cacheMaterialOnly);
    }
    if (argc >= 3 && !strcmp(argv[1], "-optixcmd")) {
        MessageBox(0, "optixcmd", "optixcmd", MB_OK);
        extern int optixcmd(const QCoreApplication& app, int port);
        int port = atoi(argv[2]);
        return optixcmd(a, port);
    }
    if (argc >= 3 && !strcmp(argv[1], "-optix")) {
        //MessageBox(0, "runner", "runner", MB_OK);
        extern int optix_main(const QCoreApplication & app, 
                            int port,
                            const char* cachedir,
                            int cachenum,
                            int sFrame,
                            int eFrame);
        int port = -1;
        if (argc >= 5 && !strcmp(argv[3], "-port"))
            port = atoi(argv[4]);
        char* cachedir = nullptr;
        int cachenum = 0, sFrame = 0, eFrame = 0;
        if (argc >= 7 && !strcmp(argv[5], "-cachedir"))
            cachedir = argv[6];
        if (argc >= 9 && !strcmp(argv[7], "-cachenum"))
            cachenum = atoi(argv[8]);
        if (argc >= 11 && !strcmp(argv[9], "-beginFrame"))
            sFrame = atoi(argv[10]);
        if (argc >= 13 && !strcmp(argv[11], "-endFrame"))
            eFrame = atoi(argv[12]);
        return optix_main(a, port, cachedir, cachenum, sFrame, eFrame);
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

    //entrance for the zenoedit-player.
    if (argc >= 2 && !strcmp(argv[1], "--record"))
    {
        extern int record_main(const QCoreApplication& app);
        return record_main(a);
    }

    QTranslator t;
    QTranslator qtTran;
    QSettings settings(zsCompanyName, zsEditor);
    QVariant use_chinese = settings.value("use_chinese");

    if (use_chinese.isNull() || use_chinese.toBool()) {
        if (t.load(":languages/zh.qm")) {
            a.installTranslator(&t);
        }
        if (qtTran.load(":languages/qt_zh_CN.qm")) {
            a.installTranslator(&qtTran);
        }
    }

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
