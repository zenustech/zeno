#pragma once

#include <zeno/core/common.h>
#include <zeno/types/ObjectDef.h>
#include <zeno/core/reflectdef.h>
#include <glm/glm.hpp>
#include "reflect/metadata.hpp"
#include "reflect/registry.hpp"
#include "reflect/container/object_proxy"
#include "reflect/container/any"
#include "reflect/container/arraylist"
#include <vector>
#include <memory>

/*
这里是直接将已定义的类型或者内置类型，直接注册反射类型，如果需要定义颜色，可以到ui\zenqt\style\colormanager.cpp上登记

还有一种定义自定义primitive类型的方式是在一个单独的文件里定义反射类，参考HeatmapObject.h里的HeatmapData2，
但这种方式的缺点是不会生成枚举值的hashcode，比如gParamType_Heatmap2这种，好处是可以统一定义颜色等属性。
*/

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
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(glm::mat3, Matrix3)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(glm::mat4, Matrix4)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<std::string>, StringList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<int>, IntList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(std::vector<float>, FloatList)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::CurvesData, Curve)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::BCurveObject, BCurve)
REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::HeatmapData, Heatmap)

REFLECT_REGISTER_RTTI_TYPE_WITH_NAME(zeno::ReflectCustomUI, ReflectCustomUI)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(zeno::ParamControl)

//只能定义基类指针
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::shared_ptr<IObject>)
REFLECT_REGISTER_RTTI_TYPE_MANUAL(std::shared_ptr<const IObject>)
//不能对子类指针实施反射，因为Any在转换的过程中无法转为基类指针，同时也得为所有子模块所有类型构造反射，不方便
//REFLECT_REGISTER_OBJECT(zeno::PrimitiveObject, Primitive)
//REFLECT_REGISTER_OBJECT(zeno::CameraObject, Camera)
//REFLECT_REGISTER_OBJECT(zeno::LightObject, Light)
//REFLECT_REGISTER_OBJECT(zeno::IObject, IObject)
//REFLECT_REGISTER_OBJECT(zeno::ListObject, List)
//REFLECT_REGISTER_OBJECT(zeno::DictObject, Dict)
//REFLECT_REGISTER_OBJECT(zeno::MeshObject, Mesh)
//REFLECT_REGISTER_OBJECT(zeno::ParticlesObject, Particles)