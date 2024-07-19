#pragma once

#include <zeno/core/common.h>
#include <zeno/core/IObject.h>

#include "reflect/core.hpp"
#include "reflect/type.hpp"
#include "reflect/metadata.hpp"
#include "reflect/registry.hpp"
#include "reflect/container/object_proxy"
#include "reflect/container/any"
#include "reflect/container/arraylist"
#include <memory>
#include "reflect/reflection.generated.hpp"



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
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::shared_ptr<zeno::IObject>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::unique_ptr<zeno::IObject>)
