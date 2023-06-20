#ifndef __ZITEM_FACTORY_H__
#define __ZITEM_FACTORY_H__

#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/comctrl/gv/callbackdef.h>

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