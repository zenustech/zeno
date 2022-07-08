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

#if 0
    QVariant scalefac_v = settings.value("scale_factor");
    if (!scalefac_v.isNull()) {
        float scalefac = scalefac_v.toFloat();
        if (scalefac >= 1.0f)
            qputenv("QT_SCALE_FACTOR", QString::number(scalefac).toLatin1());
    }
#endif

    static int calledOnce = ([]{
        zeno::getSession().eventCallbacks->triggerEvent("init");
    }(), 0);
}
