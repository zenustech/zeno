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
    else {
        return QColor();
    }
}

