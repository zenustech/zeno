#ifndef __ZITEM_FACTORY_H__
#define __ZITEM_FACTORY_H__

#include "zenoparamwidget.h"
#include "callbackdef.h"
#include <zeno/core/data.h>

namespace zenoui
{
    QGraphicsItem* createItemWidget(
        const QVariant& value,
        zeno::ParamControl ctrl,
        zeno::ParamType type,
        CallbackCollection cbSet,
        QGraphicsScene* scene,
        const zeno::ControlProperty& controlProps
    );
    extern const qreal g_ctrlHeight;
}



#endif