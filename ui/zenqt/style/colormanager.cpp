#include "colormanager.h"
#include "reflect/reflection.generated.hpp"


static std::map<size_t, std::string> g_clrMapping = {
    {zeno::reflect::get_type<bool>().type_hash(), "#FFFF00"},
    {zeno::reflect::get_type<int>().type_hash(), "#FF0000"},
    {zeno::reflect::get_type<float>().type_hash(), "#00FF00"},
    {zeno::reflect::get_type<std::string>().type_hash(), "#CCA44E"},
    {zeno::reflect::get_type<zeno::vec2i>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec2f>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec2s>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec3i>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec3f>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec3s>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec4i>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec4f>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::vec4s>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::CurvesData>().type_hash(), "#FF00FF"},
    {zeno::reflect::get_type<zeno::BCurveObject>().type_hash(), "#FF00FF"},
};


ZColorManager::ZColorManager() {
    initColorsFromCustom();
}



void ZColorManager::initColorsFromCustom()
{

}

QColor ZColorManager::getColorByType(zeno::ParamType type)
{
    auto it = g_clrMapping.find(type);
    if (it != g_clrMapping.end()) {
        return QColor(QString::fromStdString(it->second));
    }
    else {
        return QColor();
    }
}

