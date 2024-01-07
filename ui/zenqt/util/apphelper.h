#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include "zenoapplication.h"
#include "model/GraphsTreeModel.h"
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
    static QLinearGradient colorString2Grad(const QString& colorStr);
    static QString nativeWindowTitle(const QString& currentFilePath);
    static VideoRecInfo getRecordInfo(const ZENO_RECORD_RUN_INITPARAM& param);
    static QVector<QString> getKeyFrameProperty(const QVariant &val);
    static bool getCurveValue(QVariant & val);
    static bool updateCurve(QVariant oldVal, QVariant& val);
    static void dumpTabsToZsg(QDockWidget* dockWidget, RAPIDJSON_WRITER& writer);
};


#endif