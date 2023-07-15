#include <cstdio>
#include <cstring>
#include <iostream>
#include <zeno/utils/log.h>
#include <zeno/utils/Timer.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/zeno.h>
#include <string>

#include <QTcpServer>
#include <QtWidgets>
#include <QTcpSocket>

#include <zeno/utils/scope_exit.h>
#include "corelaunch.h"
#include "viewdecode.h"
#include "settings/zsettings.h"
#include "zenomainwindow.h"
#include <QApplication>
#include <QObject>


int optix_main(const QCoreApplication& app,
                int port,
                const char* cachedir,
                int cachenum,
                int sFrame,
                int eFrame,
                int finishedFrames,
                const char* sessionId)
{
    //MessageBox(0, "runner", "runner", MB_OK);           //convient to attach process by debugger, at windows.
#ifdef __linux__
    stderr = freopen("/dev/stdout", "w", stderr);
#endif
    std::cerr.rdbuf(std::cout.rdbuf());
    std::clog.rdbuf(std::cout.rdbuf());

    zeno::set_log_stream(std::clog);

    ZenoMainWindow tempWindow(nullptr, 0, PANEL_OPTIX_VIEW);
    tempWindow.showMaximized();
    tempWindow.optixClientRun(port, cachedir, cachenum, sFrame, eFrame, finishedFrames, sessionId);
    return app.exec();
}