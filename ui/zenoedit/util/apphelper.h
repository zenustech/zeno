#ifndef __ZENOEDIT_HELPER__
#define __ZENOEDIT_HELPER__

#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenomainwindow.h"
#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zenoui/comctrl/gv/zveceditoritem.h>
#include <viewport/viewportwidget.h>
#include "launch/corelaunch.h"
#include "settings/zsettings.h"

class AppHelper
{
public:
    static QModelIndexList getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& sockName, bool bInput);
    static QLinearGradient colorString2Grad(const QString& colorStr);
    static INPUT_SOCKET getInputSocket(const QPersistentModelIndex& index, const QString& inSock, bool& exist);
    static void ensureSRCDSTlastKey(INPUT_SOCKETS& inputs, OUTPUT_SOCKETS& outputs);
    static QString nativeWindowTitle(const QString& currentFilePath);
    static void socketEditFinished(QVariant newValue, QPersistentModelIndex nodeIdx, QPersistentModelIndex paramIdx);
    static void modifyLightData(QPersistentModelIndex nodeIdx);
    static void initLaunchCacheParam(LAUNCH_PARAM& param);
};


#endif