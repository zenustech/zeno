#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"
#include "zeno/utils/log.h"
#include "zeno/zeno.h"
#include "zeno/extra/EventCallbacks.h"
#include "startup/pythonenv.h"
#include "util/apiMgr.h"

int main_cmdtool(const ZenoApplication& a, int argc);
int main_cmdtool(const ZenoApplication& a, int argc)
{
    ApiMgr* apiMgr = a.apiMgr();
    if (apiMgr) {
        apiMgr->start();
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
    zeno::getSession().eventCallbacks->triggerEvent("editorConstructed");
    mainWindow.showMaximized();
    if (argc >= 3) {
        QCommandLineParser cmdParser;
        cmdParser.addHelpOption();
        cmdParser.addOptions({
            {"cmdtool", "cmdtool", "cmd tool command"},
            {"cachePath", "cachePath", "cachePath"},
            });
        cmdParser.process(a);

        QString cachapath;
        if (cmdParser.isSet("cachePath")) {
            cachapath = cmdParser.value("cachePath");
            cachapath.replace('\\', '/');
        }

        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        bool ret = mgr->initCacheDir(false, cachapath, true);
    }
    else {
        return 0;
    }

    int result = a.exec();

    if (apiMgr) {
        apiMgr->stop();
    }

    return result;
}