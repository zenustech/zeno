#pragma once

#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INode.h>
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



REFLECT_REGISTER_RTTI_TYPE_MANUAL(bool)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(int)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(float)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(double)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::string)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec2i)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec2f)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec2s)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec3i)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec3f)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec3s)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec4i)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec4f)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::vec4s)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::INode)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::vector<std::string>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::vector<int>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::vector<float>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::CurvesData)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::BCurveObject)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::shared_ptr<zeno::IObject>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::unique_ptr<zeno::IObject>)
