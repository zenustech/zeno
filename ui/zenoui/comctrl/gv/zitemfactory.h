#ifndef __ZITEM_FACTORY_H__
#define __ZITEM_FACTORY_H__

#include "zenoparamwidget.h"
#include "callbackdef.h"

namespace zenoui
{
    QGraphicsItem* createItemWidget(
        const QVariant& value,
        PARAM_CONTROL ctrl,
        const QString& type,
        CallbackCollection cbSet,
        QGraphicsScene* scene,
        const QVariant& controlProps
    );
    extern const qreal g_ctrlHeight;
}



#endif