#ifndef __COLOR_MANAGER_H__
#define __COLOR_MANAGER_H__

#include "uicommon.h"

class ZColorManager
{
public:
    ZColorManager();
    ZColorManager(const ZColorManager&) = delete;
    void initColorsFromCustom();
    static QColor getColorByType(zeno::ParamType type);
};


#endif