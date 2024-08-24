#include "colormanager.h"

ZColorManager::ZColorManager() {
    initColorsFromCustom();
}

void ZColorManager::initColorsFromCustom()
{

}

QColor ZColorManager::getColorByType(zeno::ParamType type)
{
    std::string_view color, name;
    if (zeno::getSession().getObjUIInfo(type, color, name)) {
        return QColor(QString::fromLatin1(color.data()));
    }
    else if (type == Param_Wildcard || type == Obj_Wildcard) {
        return QColor(255, 251, 240);
    }
    else {
        return QColor();
    }
}

