#ifndef __ZENO_GV_HELPER_H__
#define __ZENO_GV_HELPER_H__

#include "zgraphicslayoutitem.h"
#include "zgraphicslayout.h"

class ZenoGvHelper
{
public:
    static void setSizeInfo(QGraphicsItem* item, const SizeInfo& sz);
    static QSizeF sizehintByPolicy(QGraphicsItem* item);
};


#endif