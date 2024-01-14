#ifndef __ZENO_GV_HELPER_H__
#define __ZENO_GV_HELPER_H__

#include "zgraphicslayoutitem.h"
#include "zgraphicslayout.h"
#include "uicommon.h"

class ZenoGvHelper
{
public:
    static void setSizeInfo(QGraphicsItem* item, const SizeInfo& sz);
    static QSizeF sizehintByPolicy(QGraphicsItem* item);
    static void setValue(QGraphicsItem* item, zeno::ParamType type, const QVariant& value, QGraphicsScene* pScene);
    static void setCtrlProperties(QGraphicsItem *item,  const QVariant &value);
};


#endif