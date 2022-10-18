#ifndef __ZWIDGET_FACTORY_H__
#define __ZWIDGET_FACTORY_H__

#include <QtWidgets>
#include "gv/callbackdef.h"
#include <zenomodel/include/modeldata.h>

namespace zenoui
{
    QWidget* createWidget(
        const QVariant& value,
        PARAM_CONTROL ctrl,
        const QString& type,
        Callback_EditFinished cbFunc,
        CALLBACK_SWITCH cbSwitch,
        const QVariant& supply = QVariant()
    );
}

#endif