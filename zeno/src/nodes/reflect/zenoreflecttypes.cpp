#pragma once

#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INode.h>
#include <zeno/types/ObjectDef.h>
#include "reflect/core.hpp"
#include "reflect/type.hpp"
#include "reflect/metadata.hpp"
#include "reflect/registry.hpp"
#include "reflect/container/object_proxy"
#include "reflect/container/any"
#include "reflect/container/arraylist"
#include <memory>
#include "reflect/reflection.generated.hpp"


std::map<size_t, std::string> g_clrMapping = {
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



REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(bool, Bool)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(int, Int)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(float, Float)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(double, Double)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::string, String)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec2i, Vec2i)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec2f, Vec2f)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec2s, Vec2s)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec3i, Vec3i)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec3f, Vec3f)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec3s, Vec3s)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec4i, Vec4i)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec4f, Vec4f)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::vec4s, Vec4s)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::INode, INode)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<std::string>, StringList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<int>, IntList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<float>, FloatList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::CurvesData, Curve)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::BCurveObject, BCurve)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::HeatmapData, Heatmap)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::PrimitiveObject>, Primitive)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::CameraObject>, Camera)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::LightObject>, Light)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::IObject>, sharedIObject)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::ListObject>, List)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::shared_ptr<zeno::DictObject>, Dict)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::unique_ptr<zeno::IObject>, uniqueIObject)
