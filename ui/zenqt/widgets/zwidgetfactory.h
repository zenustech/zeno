#ifndef __ZWIDGET_FACTORY_H__
#define __ZWIDGET_FACTORY_H__

#include <QtWidgets>
#include "nodeeditor/gv/callbackdef.h"
#include <zeno/core/data.h>

namespace zenoui
{
    QWidget* createWidget(
        const QModelIndex& nodeIdx,
        const zeno::reflect::Any& value,
        zeno::ParamControl ctrl,
        const zeno::ParamType paramType,
        CallbackCollection cbSet,
        const zeno::reflect::Any& properties = zeno::reflect::Any()
    );

    bool isMatchControl(zeno::ParamControl ctrl, QWidget* pControl);
}

#endif