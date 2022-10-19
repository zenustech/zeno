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
        const CONTROL_PROPERTIES& properties = CONTROL_PROPERTIES()
    );

    bool isMatchControl(PARAM_CONTROL ctrl, QWidget* pControl);
    void updateValue(QWidget* pControl, const QVariant& value);
}

#endif