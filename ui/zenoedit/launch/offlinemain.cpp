#include "corelaunch.h"
#include "serialize.h"
#include "util/log.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/jsonhelper.h>
#include <zenomodel/include/modelrole.h>
#include <zeno/core/Session.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include "util/apphelper.h"
#include "launch/ztcpserver.h"

int offline_main(const QCoreApplication& app);
int offline_main(const QCoreApplication& app) {
    QCommandLineParser cmdParser;
    cmdParser.addHelpOption();
    cmdParser.addOptions({
        {"offline", "offline", "run offline"},
        {"zsg", "zsg", "zsg file path"},
        {"frame", "frame", "frame count"},
        {"sframe", "sframe", "start frame"},
        {"cachePath", "cachePath", "cachePath"},
        {"cacheNum", "cacheNum", "cacheNum"},
        {"cacheautorm", "cacheautoremove", "remove cache after render"},
        {"subzsg", "subgraphzsg", "subgraph zsg file path"},
        });
    cmdParser.process(app);
    if (!cmdParser.isSet("zsg") || !cmdParser.isSet("frame") || !cmdParser.isSet("sframe")) {
        zeno::log_info("missing parameter.");
        return -1;
    }
    ZENO_RECORD_RUN_INITPARAM param;
    param.sZsgPath = cmdParser.value("zsg");
    if (cmdParser.isSet("subzsg"))
    {
        param.subZsg = cmdParser.value("subzsg");
    }
    LAUNCH_PARAM launchparam;
    launchparam.beginFrame = cmdParser.value("sframe").toInt();
    launchparam.endFrame = launchparam.beginFrame + cmdParser.value("frame").toInt() - 1;
    if (cmdParser.isSet("cachePath")) {
        QString text = cmdParser.value("cachePath");
        text.replace('\\', '/');
        launchparam.cacheDir = text;
        launchparam.enableCache = true;
        launchparam.tempDir = false;
        if (!QDir(text).exists())
            QDir().mkdir(text);
        if (cmdParser.isSet("cacheautorm"))
            launchparam.autoRmCurcache = cmdParser.value("cacheautorm").toInt();
        else
            launchparam.autoRmCurcache = true;
        if (cmdParser.isSet("cacheNum"))
            launchparam.cacheNum = cmdParser.value("cacheNum").toInt();
        else
            launchparam.cacheNum = 1;
    }
    else {
        launchparam.enableCache = false;
    }

    zeno::log_info("running in offline mode, file=[{}], begin={}, end={}", launchparam.zsgPath.toStdString(), launchparam.beginFrame, launchparam.endFrame);

    ZTcpServer* server = zenoApp->getServer();
    if (server)
        QObject::connect(server, &ZTcpServer::runFinished, [=]() {
            zeno::log_info("program finished");
            QApplication::exit(0);
            });
    AppHelper::openZsgAndRun(param, launchparam);

    return app.exec();
}
