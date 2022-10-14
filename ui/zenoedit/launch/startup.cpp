#include <QtWidgets>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include "../settings/zsettings.h"
#include "startup.h"

bool initZenCache()
{
    QSettings settings(zsCompanyName, zsEditor);
    const QString& cachedir = settings.value("zencachedir").toString();
    const QString& cachenum = settings.value("zencachenum").toString();
    bool bDiskCache = false;
    int cnum = cachenum.toInt(&bDiskCache);
    bDiskCache = bDiskCache && QFileInfo(cachedir).isDir() && cnum > 0;
    if (bDiskCache) {
        auto cdir = cachedir.toStdString();
        zeno::getSession().globalComm->frameCache(cdir.c_str(), cnum);
    }
    else {
        zeno::getSession().globalComm->frameCache("", 0);
    }
    return bDiskCache;
}