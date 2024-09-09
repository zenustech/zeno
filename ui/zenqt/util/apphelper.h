#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <viewport/viewportwidget.h>
#include "settings/zsettings.h"
#include "viewport/recordvideomgr.h"
#include "panel/zenospreadsheet.h"

class AppHelper
{
public:
    static QString nativeWindowTitle(const QString& currentFilePath);
    static VideoRecInfo getRecordInfo(const ZENO_RECORD_RUN_INITPARAM& param);
    static void dumpTabsToZsg(QDockWidget* dockWidget, RAPIDJSON_WRITER& writer);
};


#endif