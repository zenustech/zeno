#include <zeno/extra/EventCallbacks.h>
#include <zeno/extra/assetDir.h>
#include <zeno/core/Session.h>
#include "zstartup.h"
#include <QSettings>

void startUp()
{
    QSettings settings("ZenusTech", "Zeno");
    QVariant nas_loc_v = settings.value("nas_loc");
    if (!nas_loc_v.isNull()) {
        zeno::setConfigVariable("NASLOC", nas_loc_v.toString().toStdString());
    }

    static int calledOnce = ([]{
        zeno::getSession().eventCallbacks->triggerEvent("init");
    }(), 0);
}
