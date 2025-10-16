#include <zeno/utils/helper.h>
#include <regex>
#include <zeno/core/CoreParam.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/core/Graph.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/ObjectDef.h>
#include <zeno/core/reflectdef.h>
#include <regex>
#include <zeno/core/typeinfo.h>
#include <reflect/type.hpp>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/interfaceutil.h>
#include "reflect/reflection.generated.hpp"


using namespace zeno::types;
using namespace zeno::reflect;

namespace zeno {

    ZENO_API ParamType convertToType(std::string const& type, const std::string_view& param_name) {
        //TODO: deprecated literal representation.
        if (type == "string" || type == "readpath" || type == "writepath" || type == "diratory" || type == "multiline_string")
        {
            return gParamType_String;
        }
        else if (type == "bool") { return gParamType_Bool; }
        else if (type == "int") { return gParamType_Int; }
        else if (type == "float") { return gParamType_Float; }
        else if (type == "NumericObject") { return gParamType_Float; }
        else if (type == "vec2i") { return gParamType_Vec2i; }
        else if (type == "vec3i") { return gParamType_Vec3i; }
        else if (type == "vec4i") { return gParamType_Vec4i; }
        else if (type == "vec2f") { return gParamType_Vec2f; }
        else if (type == "vec3f") { return gParamType_Vec3f; }
        else if (type == "vec4f") { return gParamType_Vec4f; }
        else if (type == "Matrix4") { return gParamType_Matrix4; }
        else if (type == "iobject") { return gParamType_IObject; }
        else if (type == "prim" || type == "PrimitiveObject" || type == "primitive") { return gParamType_Primitive; }
        else if (type == "geometry") { return gParamType_Geometry; }
        else if (type == "list" || type == "ListObject") { return gParamType_List; }
        else if (type == "dict" || type == "DictObject" || type == "dict") { return gParamType_Dict; }
        else if (type == "colorvec3f") { return gParamType_Vec3f; }
        else if (type == "color") { return gParamType_Heatmap; }
        else if (type == "curve") { return gParamType_Curve; }
        else if (starts_with(type, "enum ")) { return gParamType_String; }
        else if (type == "AxisObject") { return gParamType_IObject; }
        else if (type == "CameraObject") { return gParamType_Camera; }
        else if (type == "LightObject") { return gParamType_Light; }
        else if (type == "FunctionObject") { return gParamType_IObject; }
        else if (type == "object" ||
            type == "IObject" ||
            type == "zany" ||
            type == "material" ||
            type == "texture" ||
            type == "instancing" ||
            type == "shader" ||
            type == "MaterialObject" ||
            type == "LBvh") {
            return gParamType_IObject;
        }
        else if (type == "Material") {
            return gParamType_Material;
        }
        else if (type == "VDBGrid") {
            return gParamType_IObject;
        }
        else if (type == "") {
            //类型名字为空时，只能根据参数名字去猜测
            if (param_name == "prim") {
                return gParamType_Primitive;
            }
            else if (param_name == "object") {
                return gParamType_IObject;
            }
            else if (param_name == "list" || param_name == "droppedList") { return gParamType_List; }
            else if (param_name == "dict") { return gParamType_Dict; }
            else if (param_name == "camera" || param_name == "cam") {
                return gParamType_Camera;
            }
            else if (param_name == "light") {
                return gParamType_Light;
            }
            else if (param_name == "FOR" || param_name == "FUNC" || param_name == "function") {
                return gParamType_IObject;    //只能给Object了，不然就要再分配一个枚举值
            }
            else if (param_name == "true" ||
                param_name == "false" ||
                param_name == "result" ||
                param_name == "SRC" ||
                param_name == "DST" ||
                param_name == "json" ||
                param_name == "port" ||
                param_name == "data" ||
                param_name == "mtl") {
                return gParamType_IObject;
            }
            else if (param_name == "VDBGrid" || param_name == "grid") {
                return gParamType_IObject;
            }
            else if (param_name == "heatmap") {
                return gParamType_Heatmap;
            }
            else {
                return gParamType_IObject;
            }
        }
        else if (type == "null") {
            return Param_Null;
        }
        else
            return gParamType_IObject;    //zeno各个模块定义的类型不规范程度很大，而且积累了很多，很难一下子改好，所以不明类型都转成obj
    }

    ZENO_API bool isAnyEqual(const Any& lhs, const Any& rhs)
    {
        if (lhs.type() != rhs.type() || lhs.has_value() != rhs.has_value())
            return false;       //对于int和float的同等值，可能会漏

        if (!lhs.has_value())
            return true;    //null

        size_t lhsType = lhs.type().get_decayed_hash();
        if (lhsType == 0)
            lhsType = lhs.type().hash_code();

        switch (lhsType)
        {
        case gParamType_Int:    return any_cast<int>(lhs) == any_cast<int>(rhs);
        case gParamType_Float:  return any_cast<float>(lhs) == any_cast<float>(rhs);
        case gParamType_Bool:   return any_cast<bool>(lhs) == any_cast<bool>(rhs);
        case gParamType_String: return any_cast<std::string>(lhs) == any_cast<std::string>(rhs);
        case gParamType_PrimVariant:
            return any_cast<PrimVar>(lhs) == any_cast<PrimVar>(rhs);
        case gParamType_VecEdit: {
            auto vec1 = any_cast<vecvar>(lhs);
            auto vec2 = any_cast<vecvar>(rhs);
            if (vec1 != vec2) return false;
            for (int i = 0; i < vec1.size(); i++) {
                if (vec1[i] != vec2[i])
                    return false;
            }
            return true;
        }
        case gParamType_Vec2f: {
            auto vec1 = any_cast<zeno::vec2f>(lhs);
            auto vec2 = any_cast<zeno::vec2f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec2i: {
            auto vec1 = any_cast<zeno::vec2i>(lhs);
            auto vec2 = any_cast<zeno::vec2i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec2s: {
            auto vec1 = any_cast<zeno::vec2s>(lhs);
            auto vec2 = any_cast<zeno::vec2s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec3f: {
            auto vec1 = any_cast<zeno::vec3f>(lhs);
            auto vec2 = any_cast<zeno::vec3f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec3i: {
            auto vec1 = any_cast<zeno::vec3i>(lhs);
            auto vec2 = any_cast<zeno::vec3i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec3s: {
            auto vec1 = any_cast<zeno::vec3s>(lhs);
            auto vec2 = any_cast<zeno::vec3s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec4f: {
            auto vec1 = any_cast<zeno::vec4f>(lhs);
            auto vec2 = any_cast<zeno::vec4f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Vec4i: {
            auto vec1 = any_cast<zeno::vec4i>(lhs);
            auto vec2 = any_cast<zeno::vec4i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Vec4s: {
            auto vec1 = any_cast<zeno::vec4s>(lhs);
            auto vec2 = any_cast<zeno::vec4s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Curve: {
            auto curve1 = any_cast<zeno::CurvesData>(lhs);
            auto curve2 = any_cast<zeno::CurvesData>(rhs);
            return curve1 == curve2;
        }
        default:
            return lhs == rhs;
        }
    }

    std::string any_cast_to_string(const Any& value)
    {
        std::string str;
        if (value.type() == zeno::reflect::type_info<const char*>()) {
            str = zeno::reflect::any_cast<const char*>(value);
        }
        else if (value.type() == zeno::reflect::type_info<std::string>()) {
            str = zeno::reflect::any_cast<std::string>(value);
        }
        else if (value.type() == zeno::reflect::type_info<zeno::String>()) {
            str = zsString2Std(zeno::reflect::any_cast<zeno::String>(value));
        }
        else {
            throw UnimplError("any cast error, the value is not a string");
        }
        return str;
    }

    ZENO_API std::string paramTypeToString(ParamType type)
    {
        //TODO: 自定义类型的处理方式？
        switch (type)
        {
        case Param_Null:    return "null";
        case gParamType_Bool:    return "bool";
        case gParamType_Int:     return "int";
        case gParamType_String:  return "string";
        case gParamType_Float:   return "float";
        case gParamType_Vec2i:   return "vec2i";
        case gParamType_Vec3i:   return "vec3i";
        case gParamType_Vec4i:   return "vec4i";
        case gParamType_Vec2f:   return "vec2f";
        case gParamType_Vec3f:   return "vec3f";
        case gParamType_Vec4f:   return "vec4f";
        case gParamType_IObject: return "iobject";
        case gParamType_Primitive:    return "prim";
        case gParamType_Geometry:return "geometry";
        case gParamType_Dict:    return "dict";
        case gParamType_List:    return "list";
        case gParamType_Curve:   return "curve";
        case gParamType_Heatmap: return "color";
        case gParamType_AnyNumeric: return "numeric";
        case gParamType_Matrix4: return "Matrix4";
        case gParamType_Material: return "Material";
        default:
            return "";
        }
    }

    bool ListHasPrimObj(zeno::ListObject* list) {
        for (auto elem : list->get()) {
            if (dynamic_cast<PrimitiveObject*>(elem)) {
                return false;
            }
        }
        return true;
    }

    bool DictHasPrimObj(zeno::DictObject* dict) {
        for (auto& [key, elem] : dict->get()) {
            if (dynamic_cast<PrimitiveObject*>(elem)) {
                return false;
            }
        }
        return true;
    }

    ZENO_API bool convertToOriginalVar(zeno::reflect::Any& editvar, const ParamType type) {
        if (!editvar.has_value()) return false;

        ParamType anyType = editvar.type().hash_code();
        if (anyType == gParamType_PrimVariant) {
            PrimVar var = any_cast<PrimVar>(editvar);
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int> ||
                    std::is_same_v<T, float> ||
                    std::is_same_v<T, std::string>) {
                    editvar = arg;
                }
                }, var);
        }
        else if (anyType == gParamType_VecEdit) {
            //TODO
            vecvar editvec = any_cast<vecvar>(editvar);

        }
        return true;
    }

    std::vector<zany> fromZenCache(const std::string& cachedir, int frameid) {
        std::vector<zany> objs;
        std::map<std::string, zany> _objs;
        GlobalComm::fromDisk(cachedir, frameid, _objs);
        for (const auto& iter : _objs) {
            objs.push_back(iter.second->clone());
        }
        return objs;
    }

    void merge_json(nlohmann::json& target, const nlohmann::json& source) {
        for (auto it = source.begin(); it != source.end(); ++it) {
            if (target.contains(it.key()) && target[it.key()].is_object() && it.value().is_object()) {
                // 如果都是 object，递归合并
                merge_json(target[it.key()], it.value());
            }
            else {
                // 否则直接覆盖
                target[it.key()] = it.value();
            }
        }
    }

    ZENO_API bool convertToEditVar(Any& val, const ParamType type) {
        if (!val.has_value()) return false;

        ParamType anyType = val.type().hash_code();
        //部分类型可以允许k帧，公式，因此要转换为“编辑”类型
        if (anyType == gParamType_PrimVariant || anyType == gParamType_VecEdit)
            return true;

        if (type == gParamType_Int) {
            if (anyType == gParamType_String) {
                val = PrimVar(zeno::any_cast_to_string(val));
                return true;
            }
            else if (anyType == gParamType_Float || anyType == gParamType_Double) {
                val = PrimVar((int)any_cast<float>(val));
            }
            else if (anyType == gParamType_Double) {
                val = PrimVar((int)any_cast<double>(val));
            }
            else {
                val = PrimVar(any_cast<int>(val));
            }
            return true;
        }
        else if (type == gParamType_Float) {
            if (anyType == gParamType_String) {
                val = PrimVar(zeno::any_cast_to_string(val));
                return true;
            }
            else if (anyType == gParamType_Int) {
                val = PrimVar((float)any_cast<int>(val));
            }
            else if (anyType == gParamType_Double) {
                val = PrimVar((float)any_cast<double>(val));
            }
            else {
                val = PrimVar(any_cast<float>(val));
            }
            return true;
        }
        else if (type == gParamType_Vec2f) {
            auto vec2 = any_cast<vec2f>(val);
            val = vecvar{ vec2[0], vec2[1] };
            return true;
        }
        else if (type == gParamType_Vec2i) {
            auto vec2 = any_cast<vec2i>(val);
            val = vecvar{ vec2[0], vec2[1] };
            return true;
        }
        else if (type == gParamType_Vec3i) {
            auto vec3 = any_cast<vec3i>(val);
            val = vecvar{ vec3[0], vec3[1], vec3[2] };
            return true;
        }
        else if (type == gParamType_Vec3f) {
            auto vec3 = any_cast<vec3f>(val);
            val = vecvar{ vec3[0], vec3[1], vec3[2] };
            return true;
        }
        else if (type == gParamType_Vec4i) {
            auto vec4 = any_cast<vec4i>(val);
            val = vecvar{ vec4[0], vec4[1], vec4[2], vec4[3] };
            return true;
        }
        else if (type == gParamType_Vec4f) {
            auto vec4 = any_cast<vec4f>(val);
            val = vecvar{ vec4[0], vec4[1], vec4[2], vec4[3] };
            return true;
        }
        else if (type == gParamType_Shader) {
            if (anyType == gParamType_Float) {
                val = PrimVar(any_cast<float>(val));
            }
        }
        //else if (type == gParamType_AnyList) {
        //    auto vec = any_cast<std::vector<zeno::reflect::Any>>(val);
        //    if (vec.size() == 2) {

        //    }
        //    else if (vec.size() == 3) {

        //    }
        //    else if (vec.size() == 4) {

        //    }
        //}

        if (anyType != type)
            return false;

        return true;
    }

    ZENO_API Any str2any(std::string const& defl, ParamType const& type) {
        if (defl.empty())
            return initAnyDeflValue(type);
        switch (type) {
        case gParamType_String: {
            return defl;
        }
        case gParamType_Bool: {
            if (defl == "0" || defl == "false")    return false;
            if (defl == "1" || defl == "true")     return true;
            return Any();
        }
        case gParamType_Int: {
            return std::stoi(defl);
        }
        case gParamType_Float: {
            return std::stof(defl);
        }
        case gParamType_Shader: {
            return std::stof(defl); //参考ShaderBinaryMath的in1 in2参数
        }
        case gParamType_AnyNumeric: {
            return std::stof(defl);
        }
        case gParamType_Vec2i:
        case gParamType_Vec3i:
        case gParamType_Vec4i:
        {
            std::vector<int> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stoi(v));
            }
            if (gParamType_Vec2i == type) {
                return vec2i(vec[0], vec[1]);
            }
            else if (gParamType_Vec3i == type) {
                return vec3i(vec[0], vec[1], vec[2]);
            }
            else {
                return vec4i(vec[0], vec[1], vec[2], vec[3]);
            }
            return Any();
        }
        case gParamType_Vec2f:
        case gParamType_Vec3f:
        case gParamType_Vec4f:
        {
            std::vector<float> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stof(v));
            }

            if (gParamType_Vec2f == type) {
                if (vec.size() != 2)
                    return vec2f();
                return vec2f(vec[0], vec[1]);
            }
            else if (gParamType_Vec3f == type) {
                if (vec.size() != 3)
                    return vec3f();
                return vec3f(vec[0], vec[1], vec[2]);
            }
            else {
                if (vec.size() != 4)
                    return vec4f();
                return vec4f(vec[0], vec[1], vec[2], vec[3]);
            }
            return Any();
        }
        default:
            return defl;
        }
    }

    ZENO_API zvariant str2var(std::string const& defl, ParamType const& type) {
        if (defl.empty())
            return initDeflValue(type);
        switch (type) {
        case gParamType_String: {
            return defl;
        }
        case gParamType_Bool: {
            if (defl == "0" || defl == "false")    return 0;
            if (defl == "1" || defl == "true")     return 1;
            return zvariant();
        }
        case gParamType_Int: {
            return std::stoi(defl);
        }
        case gParamType_Float: {
            return std::stof(defl);
        }
        case gParamType_Vec2i:
        case gParamType_Vec3i:
        case gParamType_Vec4i:
        {
            std::vector<int> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stoi(v));
            }
            if (gParamType_Vec2i == type) {
                return vec2i(vec[0], vec[1]);
            }
            else if (gParamType_Vec3i == type) {
                return vec3i(vec[0], vec[1], vec[2]);
            }
            else {
                return vec4i(vec[0], vec[1], vec[2], vec[3]);
            }
            return zvariant();
        }
        case gParamType_Vec2f:
        case gParamType_Vec3f:
        case gParamType_Vec4f:
        {
            std::vector<float> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stof(v));
            }

            if (gParamType_Vec2f == type) {
                if (vec.size() != 2)
                    return vec2f();
                return vec2f(vec[0], vec[1]);
            }
            else if (gParamType_Vec3f == type) {
                if (vec.size() != 3)
                    return vec3f();
                return vec3f(vec[0], vec[1], vec[2]);
            }
            else {
                if (vec.size() != 4)
                    return vec4f();
                return vec4f(vec[0], vec[1], vec[2], vec[3]);
            }
            return zvariant();
        }
        default:
            return defl;
        }
    }

    NumericValue AnyToNumeric(const zeno::reflect::Any& any, bool& bSucceed) {
        if (!any.has_value()) {
            bSucceed = false;
            return 0;
        }
        bSucceed = true;
        ParamType anyType = any.type().hash_code();
        switch (anyType)
        {
        case gParamType_Int:    return any_cast<int>(any);
        case gParamType_Float:  return any_cast<float>(any);
        case gParamType_Vec2i:  return any_cast<vec2i>(any);
        case gParamType_Vec2f:  return any_cast<vec2f>(any);
        case gParamType_Vec3i:  return any_cast<vec3i>(any);
        case gParamType_Vec3f:  return any_cast<vec3f>(any);
        case gParamType_Vec4i:  return any_cast<vec4i>(any);
        case gParamType_Vec4f:  return any_cast<vec4f>(any);
        case gParamType_PrimVariant:
        case gParamType_VecEdit:
        {
            //不考虑编辑类型，因为这是由节点内部处理的数据，我们只要取result就行
        }
        default:
        {
            bSucceed = false;
            return 0;
        }
        }
    }

    ZENO_API zvariant AnyToZVariant(Any const& var) {
        if (!var.has_value())
            return zvariant();
        if (get_type<int>() == var.type()) {
            return any_cast<int>(var);
        }
        else if (get_type<float>() == var.type()) {
            return any_cast<float>(var);
        }
        else if (get_type<std::string>() == var.type()) {
            return zeno::any_cast_to_string(var);
        }
        else if (get_type<zeno::vec2i>() == var.type()) {
            return any_cast<zeno::vec2i>(var);
        }
        else if (get_type<zeno::vec3i>() == var.type()) {
            return any_cast<zeno::vec3i>(var);
        }
        else if (get_type<zeno::vec4i>() == var.type()) {
            return any_cast<zeno::vec4i>(var);
        }
        else if (get_type<zeno::vec2f>() == var.type()) {
            return any_cast<zeno::vec2f>(var);
        }
        else if (get_type<zeno::vec3f>() == var.type()) {
            return any_cast<zeno::vec3f>(var);
        }
        else if (get_type<zeno::vec4f>() == var.type()) {
            return any_cast<zeno::vec4f>(var);
        }
        else if (get_type<zeno::vec2s>() == var.type()) {
            return any_cast<zeno::vec2s>(var);
        }
        else if (get_type<zeno::vec3s>() == var.type()) {
            return any_cast<zeno::vec3s>(var);
        }
        else if (get_type<zeno::vec4s>() == var.type()) {
            return any_cast<zeno::vec4s>(var);
        }
        else if (get_type<zeno::PrimVar>() == var.type()) {
            zeno::PrimVar primvar = any_cast<zeno::PrimVar>(var);
            return std::visit([](zeno::PrimVar&& pvar) -> zvariant {
                using T = std::decay_t<decltype(pvar)>;
                if constexpr (std::is_same_v<int, T>) {
                    return std::get<int>(pvar);
                }
                else if constexpr (std::is_same_v<float, T>) {
                    return std::get<float>(pvar);
                }
                else if constexpr (std::is_same_v<std::string, T>) {
                    return std::get<std::string>(pvar);
                }
                else {
                    return zvariant();
                }
                }, primvar);
        }
        else
            return zvariant();
    }

    ZENO_API Any initAnyDeflValue(ParamType const& type)
    {
        if (type == gParamType_String || type == gParamType_PrimVariant) {
            return std::string("");     //要注意和char*常量区分，any::get_type的时候是不一样的
        }
        else if (type == gParamType_Float)
        {
            return (float)0.;
        }
        else if (type == gParamType_Int)
        {
            return (int)0;
        }
        else if (type == gParamType_Bool)
        {
            return false;
        }
        else if (type == gParamType_Vec2i)
        {
            return vec2i();
        }
        else if (type == gParamType_Vec2f)
        {
            return vec2f();
        }
        else if (type == gParamType_Vec3i)
        {
            return vec3i();
        }
        else if (type == gParamType_Vec3f)
        {
            return vec3f();
        }
        else if (type == gParamType_Vec4i)
        {
            return vec4i();
        }
        else if (type == gParamType_Vec4f)
        {
            return vec4f();
        }
        else if (type == gParamType_Matrix4)
        {
            return glm::mat4(1.0);
        }
        else if (type == gParamType_Matrix3)
        {
            return glm::mat3(1.0);
        }
        else if (type == gParamType_GLMVec3)
        {
            return glm::vec3(0);
        }
        else if (type == gParamType_Curve)
        {
            return make_any<CurvesData>();
        }
        else if (type == gParamType_IObject)
        {
            return nullptr;
        }
        else if (type == gParamType_AnyNumeric)
        {
            return 0.f;
        }
        else if (type == gParamType_Heatmap)
        {
            return HeatmapData();
        }
        else if (type == gParamType_Shader)
        {
            return ShaderData();
        }
        else if (type == gParamType_StringList)
        {
            return std::vector<std::string>();
        }
        else if (type == gParamType_ListOfMat4)
        {
            return std::vector<glm::mat4>();
        }
        else if (type == gParamType_IntList)
        {
            return std::vector<int>();
        }
        else {
            assert(false);
        }
        return Any();
    }

    zvariant initDeflValue(ParamType type)
    {
        if (type == gParamType_String) {
            return "";
        }
        else if (type == gParamType_Float)
        {
            return (float)0.;
        }
        else if (type == gParamType_Int)
        {
            return (int)0;
        }
        else if (type == gParamType_Bool)
        {
            return false;
        }
        else if (type == gParamType_Vec2i)
        {
            return vec2i();
        }
        else if (type == gParamType_Vec2f)
        {
            return vec2f();
        }
        else if (type == gParamType_Vec3i)
        {
            return vec3i();
        }
        else if (type == gParamType_Vec3f)
        {
            return vec3f();
        }
        else if (type == gParamType_Vec4i)
        {
            return vec4i();
        }
        else if (type == gParamType_Vec4f)
        {
            return vec4f();
        }
        else if (type == gParamType_Curve)
        {
            return "{}";
        }
        else if (type == gParamType_Heatmap)
        {
            return "{\"nres\":1024, \"color\":\"\"}";
        }
        return zvariant();
    }

    EdgeInfo getEdgeInfo(std::shared_ptr<ObjectLink> spLink) {
        EdgeInfo edge;
        auto spOutParam = spLink->fromparam;
        auto spInParam = spLink->toparam;
        if (!spOutParam || !spInParam)
            return edge;

        auto spOutNode = spOutParam->m_wpNode;
        auto spInNode = spInParam->m_wpNode;
        if (!spOutNode || !spInNode)
            return edge;

        const std::string& outNode = spOutNode->get_name();
        const std::string& outParam = spOutParam->name;
        const std::string& inNode = spInNode->get_name();
        const std::string& inParam = spInParam->name;
        edge = { outNode, outParam, spLink->fromkey, inNode, inParam, spLink->tokey, spLink->targetParam, true };
        return edge;
    }

    EdgeInfo getEdgeInfo(std::shared_ptr<PrimitiveLink> spLink) {
        EdgeInfo edge;
        auto spOutParam = spLink->fromparam;
        auto spInParam = spLink->toparam;
        if (!spOutParam || !spInParam)
            return edge;

        auto spOutNode = spOutParam->m_wpNode;
        auto spInNode = spInParam->m_wpNode;
        if (!spOutNode || !spInNode)
            return edge;

        const std::string& outNode = spOutNode->get_name();
        const std::string& outParam = spOutParam->name;
        const std::string& inNode = spInNode->get_name();
        const std::string& inParam = spInParam->name;
        edge = { outNode, outParam, spLink->fromkey, inNode, inParam, spLink->tokey, spLink->targetParam, false };
        return edge;
    }

    std::string generateObjKey(std::shared_ptr<IObject> spObject) {
        return "";    //TODO
    }

    std::string uniqueName(std::string prefix, std::set<std::string> existing) {
        std::string param_name = prefix;
        int idx = 1;
        while (existing.find(param_name) != existing.end()) {
            param_name = prefix + std::to_string(idx++);
        }
        return param_name;
    }

    ZENO_API std::string objPathToStr(ObjPath path) {
        return path;
    }

    ObjPath strToObjPath(const std::string& str)
    {
        return str;
    }

    PrimitiveParams customUiToParams(const CustomUIParams& customparams) {
        PrimitiveParams params;
        for (auto tab : customparams) {
            for (auto group : tab.groups) {
                params.insert(params.end(), group.params.begin(), group.params.end());
            }
        }
        return params;
    }

    ZENO_API void parseUpdateInfo(const CustomUI& customui, ParamsUpdateInfo& infos)
    {
        for (const zeno::ParamTab& tab : customui.inputPrims)
        {
            for (const zeno::ParamGroup& group : tab.groups)
            {
                for (const zeno::ParamPrimitive& param : group.params)
                {
                    zeno::ParamPrimitive info;
                    info.bInput = true;
                    info.control = param.control;
                    info.type = param.type;
                    info.defl = param.defl;
                    info.name = param.name;
                    info.tooltip = param.tooltip;
                    info.socketType = param.socketType;
                    info.ctrlProps = param.ctrlProps;
                    infos.push_back({ info, "" });
                }
            }
        }
        for (const zeno::ParamPrimitive& param : customui.outputPrims)
        {
            zeno::ParamPrimitive info;
            info.bInput = false;
            info.control = param.control;
            info.type = param.type;
            info.defl = param.defl;
            info.name = param.name;
            info.tooltip = param.tooltip;
            info.socketType = param.socketType;
            info.ctrlProps = param.ctrlProps;
            infos.push_back({ info, "" });
        }
        for (const zeno::ParamObject& param : customui.inputObjs)
        {
            zeno::ParamObject info;
            info.bInput = true;
            info.type = param.type;
            info.name = param.name;
            info.tooltip = param.tooltip;
            info.socketType = param.socketType;
            infos.push_back({ info, "" });
        }
        for (const zeno::ParamObject& param : customui.outputObjs)
        {
            zeno::ParamObject info;
            info.bInput = false;
            info.type = param.type;
            info.name = param.name;
            info.tooltip = param.tooltip;
            info.socketType = param.socketType;
            infos.push_back({ info, "" });
        }
    }

    void initControlsByType(CustomUI& ui) {
        for (ParamTab& tab : ui.inputPrims)
        {
            for (ParamGroup& group : tab.groups)
            {
                for (ParamPrimitive& param : group.params)
                {
                    if (param.type != Param_Null && param.control == NullControl)
                        param.control = getDefaultControl(param.type);
                }
            }
        }
    }

    std::set<std::string> getReferPaths(const zvariant& val)
    {
        return std::visit([](const auto& arg)->std::set<std::string> {
            using T = std::decay_t<decltype(arg)>;
            std::set<std::string> paths;
            if constexpr (std::is_same_v<T, std::string>) {
                paths = getReferPath(arg);
            }
            else if constexpr (std::is_same_v<T, zeno::vec2s> || std::is_same_v<T, zeno::vec3s> || std::is_same_v<T, zeno::vec4s>)
            {
                for (int i = 0; i < arg.size(); i++)
                {
                    auto res = getReferPath(arg[i]);
                    paths.insert(res.begin(), res.end());
                }
            }
            return paths;
            }, val);
    }

    formula_tip_info getNodesByPath(const std::string& nodeabspath, const std::string& graphpath, const std::string& node_part)
    {
        formula_tip_info ret;
        ret.type = FMLA_NO_MATCH;
        if (graphpath.empty())
            return ret;

        std::string fullabspath = graphpath;

        if (graphpath.front() == '.' || graphpath.front() == '..') {
            auto graphparts = split_str(graphpath, '/', false);
            auto nodeparts = split_str(nodeabspath, '/', false);
            nodeparts.pop_back();   //先去掉节点本身，剩下的就是节点所处的图的绝对路径。
            nodeparts.insert(nodeparts.end(), graphparts.begin(), graphparts.end());
            fullabspath = '/' + join_str(nodeparts, '/');
        }
        else if (graphpath.front() == '/') {
            fullabspath = graphpath;
        }
        else {
            return ret;
        }

        auto spGraph = zeno::getSession().getGraphByPath(fullabspath);
        if (!spGraph)
            return ret;

        if (node_part == ".") {
            ret.ref_candidates.clear();
            ret.type = FMLA_NO_MATCH;
            return ret;
        }

        //node_part like `Cube1.position.x`
        std::vector<std::string> node_items = split_str(node_part, '.');
        bool bEndsWithDot = !node_part.empty() && node_part.back() == '.';
        const std::string& node_name = !node_items.empty() ? node_items[0] : "";
        const std::string& param_part = node_items.size() >= 2 ? node_items[1] : "";
        const std::string& param_component = node_items.size() >= 3 ? node_items[2] : "";

        std::map<std::string, NodeImpl*> nodes = spGraph->getNodes();
        for (auto& [name, spNode] : nodes) {
            if (name.find(node_name) != std::string::npos) {
                if (!param_part.empty() || (param_part.empty() && bEndsWithDot)) {
                    PrimitiveParams params = spNode->get_input_primitive_params();
                    for (auto param : params) {
                        if (param.name.find(param_part) != std::string::npos) {
                            if (!param_part.empty() && (!param_component.empty() || bEndsWithDot)) {
                                if (param.name == param_part) {
                                    switch (param.type) {
                                    case gParamType_Vec2f:
                                    case gParamType_Vec2i:
                                    {
                                        ret.type = FMLA_TIP_REFERENCE;
                                        ret.prefix = param_component;
                                        if (param_component == "x" || param_component == "y") {
                                            std::string iconres = "";
                                            ret.ref_candidates.push_back({ param_component, iconres });
                                        }
                                        else if (param_component.empty()) {
                                            ret.ref_candidates = { {"x", ""}, {"y", ""} };
                                        }
                                        else
                                        {
                                            ret.ref_candidates.clear();
                                            ret.type = FMLA_NO_MATCH;
                                            return ret;
                                        }
                                        break;
                                    }
                                    case gParamType_Vec3f:
                                    case gParamType_Vec3i:
                                    {
                                        ret.type = FMLA_TIP_REFERENCE;
                                        ret.prefix = param_component;
                                        if (param_component == "x" || param_component == "y" || param_component == "z") {
                                            std::string iconres = "";
                                            ret.ref_candidates.push_back({ param_component, iconres });
                                        }
                                        else if (param_component.empty()) {
                                            ret.ref_candidates = { {"x", ""}, {"y", ""}, {"z", ""} };
                                        }
                                        else
                                        {
                                            ret.ref_candidates.clear();
                                            ret.type = FMLA_NO_MATCH;
                                            return ret;
                                        }
                                        break;
                                    }
                                    case gParamType_Vec4f:
                                    case gParamType_Vec4i:
                                    {
                                        ret.type = FMLA_TIP_REFERENCE;
                                        ret.prefix = param_component;
                                        if (param_component == "x" || param_component == "y" || param_component == "z" || param_component == "w") {
                                            std::string iconres = "";
                                            ret.ref_candidates.push_back({ param_component, iconres });
                                        }
                                        else if (param_component.empty()) {
                                            ret.ref_candidates = { {"x", ""}, {"y", ""}, {"z", ""}, {"w", ""} };
                                        }
                                        else
                                        {
                                            ret.ref_candidates.clear();
                                            ret.type = FMLA_NO_MATCH;
                                            return ret;
                                        }
                                        break;
                                    }
                                    default:
                                    {
                                        ret.ref_candidates.clear();
                                        ret.type = FMLA_NO_MATCH;
                                        return ret;
                                    }
                                    }
                                }
                                else {
                                    ret.ref_candidates.clear();
                                    ret.type = FMLA_NO_MATCH;
                                    return ret;
                                }
                            }
                            else {
                                ret.type = FMLA_TIP_REFERENCE;
                                ret.prefix = param_part;
                                ret.ref_candidates.push_back({ param.name, "" /*icon*/ });
                            }
                        }
                    }
                }
                else {
                    ret.type = FMLA_TIP_REFERENCE;
                    ret.prefix = node_name;
                    ret.ref_candidates.push_back({ name, "" });
                }
            }
        }
        return ret;
    }

    std::string absolutePath(std::string currentPath, const std::string& path)
    {
        if (!zeno::starts_with(path, "./") && !zeno::starts_with(path, "../"))
            return path;
        if (starts_with(path, "./"))
            return currentPath + path.substr(1, path.size() - 1);
        auto vec = split_str(path, '/');
        std::string tmpPath;
        if (zeno::ends_with(currentPath, "/"))
            currentPath = currentPath.substr(0, currentPath.size() - 1);
        for (int i = vec.size() - 1; i >= 0; i--)
        {
            if (vec[i] == "..")
            {
                currentPath = currentPath.substr(0, currentPath.find_last_of("/"));
            }
            else
            {
                if (tmpPath.empty())
                    tmpPath = vec[i];
                else
                    tmpPath = vec[i] + "/" + tmpPath;
            }
        }
        return currentPath + "/" + tmpPath;
    }

    std::string relativePath(std::string currentPath, const std::string& path)
    {
        if (path.find(currentPath) != std::string::npos)
        {
            std::regex pattern(currentPath);
            return std::regex_replace(path, pattern, ".");
        }
        std::string str;
        if (zeno::ends_with(currentPath, "/"))
            currentPath = currentPath.substr(0, currentPath.size() - 1);
        int pos = currentPath.find_last_of("/");
        while (pos != std::string::npos)
        {
            if (path.find(currentPath) == std::string::npos)
            {
                if (str.empty())
                    str = "..";
                else
                    str = "../" + str;

                currentPath = currentPath.substr(0, pos);
                pos = currentPath.find_last_of("/");
            }
            else
            {
                break;
            }
        }
        std::regex regx(currentPath);
        return std::regex_replace(path, regx, str);
    }

    std::set<std::string> getReferPath(const std::string& path)
    {
        std::regex words_regex(".*ref\\(\"(.*)\"\\).*");
        std::set<std::string> result;
        std::string str = path;
        std::smatch match;
        while (std::regex_match(str, match, words_regex))
        {
            std::string val = match[1].str();
            std::regex rgx("(\\.x|\\.y|\\.z|\\.w)$");
            std::string newVal = std::regex_replace(val, rgx, "");
            result.emplace(newVal);
            val = "ref(\"" + val + "\")";
            str.replace(str.find(val), val.size(), "");
        }
        return result;
    }

    bool isObjectType(const RTTITypeInfo& type, bool& isConstPtr)
    {
        //目前TF_IsObject只是标识Object子类，不包括IObject，如果需要后者，可以判断TF_IsIObject.
        //const 也可以用generator搞
        isConstPtr = type.has_flags(TF_IsConst);
        return type.has_flags(TF_IsObject);
    }

    bool isObjectType(ParamType type)
    {
        const RTTITypeInfo& typeInfo = ReflectionRegistry::get().getRttiMap()->get(type);
        assert(typeInfo.hash_code());
        std::string rttiname(typeInfo.name());
        std::regex pattern(R"(std::shared_ptr\s*<\s*struct\s*zeno::.+Object\s*>)");
        if (std::regex_search(rttiname, pattern)) {
            return true;
        }
        return false;
    }

    bool isNumericType(ParamType type)
    {
        if (type == types::gParamType_Int || type == types::gParamType_Float || type == types::gParamType_Bool)
            return true;
        return false;
    }

    bool isNumericVecType(ParamType type)
    {
        if (isNumericType(type))
            return true;
        else if (type == types::gParamType_Vec2f || type == types::gParamType_Vec2i ||
            type == types::gParamType_Vec3f || type == types::gParamType_Vec3i ||
            type == types::gParamType_Vec4f || type == types::gParamType_Vec4i)
            return true;
        return false;
    }

    bool isSameDimensionNumericVecType(ParamType left, ParamType right)
    {
        if (left == right) {
            if (left == types::gParamType_Vec2f || left == types::gParamType_Vec2i ||
                left == types::gParamType_Vec3f || left == types::gParamType_Vec3i ||
                left == types::gParamType_Vec4f || left == types::gParamType_Vec4i)
                return true;
        } else {
            if (left == types::gParamType_Vec2i && right == types::gParamType_Vec2f || left == types::gParamType_Vec2f && right == types::gParamType_Vec2i ||
                left == types::gParamType_Vec3i && right == types::gParamType_Vec3f || left == types::gParamType_Vec3f && right == types::gParamType_Vec3i ||
                left == types::gParamType_Vec4i && right == types::gParamType_Vec4f || left == types::gParamType_Vec4f && right == types::gParamType_Vec4i)
                return true;
        }
        return false;
    }

    ZENO_API bool outParamTypeCanConvertInParamType(ParamType outType, ParamType inType, NodeDataGroup outGroup, NodeDataGroup inGroup)
    {
        if (isNumericType(outType) && isNumericVecType(inType)) {   //数值连数值vec
            return true;
        }
        else if (isSameDimensionNumericVecType(outType, inType)) { //同维度数值vec互连
            return true;
        }
        else if (outType == gParamType_Shader &&
            (inType == gParamType_Shader || isNumericType(inType) || isNumericVecType(inType)))
        {
            return true;
        }
        else if (inType == gParamType_Dict || inType == gParamType_List) {
            return true;
        }
        else if (inType == gParamType_AnyNumeric || outType == gParamType_AnyNumeric) {
            return true;
        }
        else if (inType == gParamType_ListOfMat4 || outType == gParamType_Matrix4) {
            return true;
        }
        else if (gParamType_IObject == inType && outGroup == Role_OutputObject) {    //outType的Obj类型可以转IObject
            return true;
        }
        else if (gParamType_IObject == outType && inGroup == Role_InputObject) {
            //由于一些特殊节点，比如foreachbegin，出来的是IObject，但需要连到其他的特定object节点，
            //同时又不能用wildcard（因为要和foreachend对应），这种情况允许向下转换，错误让节点自己报
            return true;
        }
        else {
            return false;
        }
    }

    ZENO_API bool isPrimVarType(ParamType type)
    {
        return type == gParamType_Int || type == gParamType_Float || type == gParamType_String || type == gParamType_Curve;
    }

    ZENO_API zeno::reflect::Any convertNumericAnyType(ParamType outType, ParamType inType, reflect::Any outputVal)
    {
        zeno::reflect::Any val;
        if (outType == types::gParamType_Int) {
            if (inType == types::gParamType_Float) {
                val = zeno::reflect::make_any<float>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec2i) {
                val = zeno::reflect::make_any<zeno::vec2i>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec2f) {
                val = zeno::reflect::make_any<zeno::vec2f>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec3i) {
                val = zeno::reflect::make_any<zeno::vec3i>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec3f) {
                val = zeno::reflect::make_any<zeno::vec3f>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec4i) {
                val = zeno::reflect::make_any<zeno::vec4i>(zeno::reflect::any_cast<int>(outputVal));
            } else if (inType == types::gParamType_Vec4f) {
                val = zeno::reflect::make_any<zeno::vec4f>(zeno::reflect::any_cast<int>(outputVal));
            } else {
                return val;
            }
        } else if (outType == types::gParamType_Float) {
            if (inType == types::gParamType_Int) {
                val = zeno::reflect::make_any<int>(zeno::reflect::any_cast<float>(outputVal));
            } else if (inType == types::gParamType_Vec2i) {
                val = zeno::reflect::make_any<zeno::vec2i>(zeno::reflect::any_cast<float>(outputVal));
            }else if (inType == types::gParamType_Vec2f) {
                val = zeno::reflect::make_any<zeno::vec2f>(zeno::reflect::any_cast<float>(outputVal));
            }else if (inType == types::gParamType_Vec3i) {
                val = zeno::reflect::make_any<zeno::vec3i>(zeno::reflect::any_cast<float>(outputVal));
            }else if (inType == types::gParamType_Vec3f) {
                val = zeno::reflect::make_any<zeno::vec3f>(zeno::reflect::any_cast<float>(outputVal));
            }else if (inType == types::gParamType_Vec4i) {
                val = zeno::reflect::make_any<zeno::vec4i>(zeno::reflect::any_cast<float>(outputVal));
            }else if (inType == types::gParamType_Vec4f) {
                val = zeno::reflect::make_any<zeno::vec4f>(zeno::reflect::any_cast<float>(outputVal));
            } else {
                return val;
            }
        } else if (outType == types::gParamType_Vec2i && inType == types::gParamType_Vec2f) {
            val = zeno::reflect::make_any<zeno::vec2f>(zeno::reflect::any_cast<zeno::vec2i>(outputVal));
        } else if (outType == types::gParamType_Vec2f && inType == types::gParamType_Vec2i) {
            val = zeno::reflect::make_any<zeno::vec2i>(zeno::reflect::any_cast<zeno::vec2f>(outputVal));
        } else if (outType == types::gParamType_Vec3i && inType == types::gParamType_Vec3f) {
            val = zeno::reflect::make_any<zeno::vec3f>(zeno::reflect::any_cast<zeno::vec3i>(outputVal));
        } else if (outType == types::gParamType_Vec3f && inType == types::gParamType_Vec3i) {
            val = zeno::reflect::make_any<zeno::vec3i>(zeno::reflect::any_cast<zeno::vec3f>(outputVal));
        } else if (outType == types::gParamType_Vec4i && inType == types::gParamType_Vec4f) {
            val = zeno::reflect::make_any<zeno::vec4f>(zeno::reflect::any_cast<zeno::vec4i>(outputVal));
        } else if (outType == types::gParamType_Vec4f && inType == types::gParamType_Vec4i) {
            val = zeno::reflect::make_any<zeno::vec4i>(zeno::reflect::any_cast<zeno::vec4f>(outputVal));
        } else {
            return val;
        }
        return val;
    }

    void getNameMappingFromReflectUI(
        reflect::TypeBase* typeBase,
        NodeImpl* node,
        std::map<std::string, std::string>& inputParams,
        std::vector<std::string>& outputParams
    )
    {
#if 0
        if (!typeBase || !node) {
            return;
        }
        for (IMemberField* field : typeBase->get_member_fields()) {
            if (field->get_field_type() == get_type<ReflectCustomUI>()) {
                Any reflectCustomUiAny = field->get_field_value(node);
                if (reflectCustomUiAny.has_value()) {
                    ReflectCustomUI reflectCustomUi = any_cast<ReflectCustomUI>(reflectCustomUiAny);
                    for (_CommonParam& param : reflectCustomUi.inputParams) {
                        std::visit([&](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, ParamObject>) {
                                inputParams.insert(std::make_pair(param.mapTo, arg.name));
                            }
                            else if constexpr (std::is_same_v<T, ParamPrimitive>) {
                                inputParams.insert(std::make_pair(param.mapTo, arg.name));
                            }
                        }, param.param);
                    }
                    for (_CommonParam& param : reflectCustomUi.outputParams) {
                        std::visit([&](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, ParamObject>) {
                                outputParams.push_back(arg.name);
                            }
                            else if constexpr (std::is_same_v<T, ParamPrimitive>) {
                                outputParams.push_back(arg.name);
                            }
                        }, param.param);
                    }
                }
                break;
            }
        }
#endif
    }

    ZENO_API bool isDerivedFromSubnetNodeName(const std::string& clsname)
    {
        if (clsname == "Subnet" || clsname == "DopNetwork") {
            return true;
        }
        return false;
    }

    ZENO_API zeno::SubnetNode* getSubnetNode(zeno::NodeImpl* pAdapter) {
        if (!pAdapter) return nullptr;
        return dynamic_cast<zeno::SubnetNode*>(pAdapter);
    }

    void propagateDirty(NodeImpl* spCurrNode, std::string varName)
    {
        std::set<ObjPath> upstreamDepNodes;
        std::set<ObjPath> upstreams;
        if (spCurrNode) {
            getUpstreamNodes(spCurrNode, upstreamDepNodes, upstreams);
            for (auto& objPath : upstreamDepNodes) {
                if (auto node = zeno::getSession().getNodeByUuidPath(objPath)) {
                    mark_dirty_by_dependNodes(node, true, upstreams);
                }
            }
        }
    }

    void getUpstreamNodes(NodeImpl* spCurrNode, std::set<ObjPath>& upstreamDepNodes, std::set<ObjPath>& upstreams, std::string outParamName)
    {
        if (!spCurrNode)
            return;
        if (auto spGraph = spCurrNode->getGraph())
        {
            spGraph->isFrameNode(spCurrNode->get_uuid());
            upstreamDepNodes.insert(spCurrNode->get_uuid_path());
        }

        if (upstreams.find(spCurrNode->get_uuid_path()) != upstreams.end()) {
            return;
        }
        if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(spCurrNode))
        {
            auto pImpl = pSubnetNode;
            auto suboutoutGetUpstreamFunc = [&pSubnetNode, &upstreamDepNodes, &upstreams](std::string paramName) {
                if (auto suboutput = pSubnetNode->get_subgraph()->getNode(paramName)) {
                    getUpstreamNodes(suboutput, upstreamDepNodes, upstreams);
                    upstreams.insert(suboutput->get_uuid_path());
                }
            };
            if (outParamName != "") {
                suboutoutGetUpstreamFunc(outParamName);
            }
            else {
                for (auto& param : pImpl->get_output_primitive_params()) {
                    suboutoutGetUpstreamFunc(param.name);
                }
                for (auto& param : pImpl->get_output_object_params()) {
                    suboutoutGetUpstreamFunc(param.name);
                }
            }
            upstreams.insert(pImpl->get_uuid_path());
        }
        else {
            auto spGraph = spCurrNode->getGraph();
            for (auto& param : spCurrNode->get_input_primitive_params()) {
                for (auto link : param.links) {
                    if (spGraph)
                    {
                        auto outParam = link.outParam;
                        auto outNode = spGraph->getNode(link.outNode);
                        assert(outNode);
                        getUpstreamNodes(outNode, upstreamDepNodes, upstreams, outParam);
                        upstreams.insert(outNode->get_uuid_path());
                    }
                }
            }
            for (auto& param : spCurrNode->get_input_object_params()) {
                for (auto link : param.links) {
                    if (spGraph)
                    {
                        auto outParam = link.outParam;
                        auto outNode = spGraph->getNode(link.outNode);
                        assert(outNode);
                        getUpstreamNodes(outNode, upstreams, upstreamDepNodes, outParam);
                        upstreams.insert(outNode->get_uuid_path());
                    }
                }
            }
            upstreams.insert(spCurrNode->get_uuid_path());
        }
        auto spGraph = spCurrNode->getGraph();
        assert(spGraph);
        if (spGraph->getParentSubnetNode() && spCurrNode->get_nodecls() == "SubInput")
        {
            auto parentSubgNode = spGraph->getParentSubnetNode();
            upstreams.insert(parentSubgNode->get_uuid_path());
            auto parentSubgNodeGetUpstreamFunc = [ &upstreams, &upstreamDepNodes, &parentSubgNode](std::string outNode, std::string outParam) {
                if (auto graph = parentSubgNode->getThisGraph()) {
                    auto node = graph->getNode(outNode);
                    assert(node);
                    getUpstreamNodes(node, upstreams, upstreamDepNodes, outParam);
                    upstreams.insert(node->get_uuid_path());
                }
            };
            bool find = false;
            const auto& parentSubgNodePrimsInput = parentSubgNode->get_input_prim_param(spCurrNode->get_name(), &find);
            if (find) {
                for (auto link : parentSubgNodePrimsInput.links) {
                    parentSubgNodeGetUpstreamFunc(link.outNode, link.outParam);
                }
            }
            bool find2 = false;
            const auto& parentSubgNodeObjsInput = parentSubgNode->get_input_obj_param(spCurrNode->get_name(), &find2);
            if (find2) {
                for (auto link : parentSubgNodeObjsInput.links) {
                    parentSubgNodeGetUpstreamFunc(link.outNode, link.outParam);
                }
            }
        }
    }

    void mark_dirty_by_dependNodes(NodeImpl* spCurrNode, bool bOn, std::set<ObjPath> nodesRange, std::string inParamName /*= ""*/)
    {
        if (!nodesRange.empty()) {
            if (nodesRange.find(spCurrNode->get_uuid_path()) == nodesRange.end()) {
                return;
            }
        }

        if (spCurrNode->is_dirty())
            return;
        spCurrNode->mark_dirty(true, Dirty_All, true, false);

        if (bOn) {
            auto spGraph = spCurrNode->getGraph();
            for (auto& param : spCurrNode->get_output_primitive_params()) {
                for (auto link : param.links) {
                    if (spGraph) {
                        auto inParam = link.inParam;
                        auto inNode = spGraph->getNode(link.inNode);
                        assert(inNode);
                        mark_dirty_by_dependNodes(inNode, bOn, nodesRange, inParam);
                    }
                }
            }
            for (auto& param : spCurrNode->get_output_object_params()) {
                for (auto link : param.links) {
                    if (spGraph) {
                        auto inParam = link.inParam;
                        auto inNode = spGraph->getNode(link.inNode);
                        assert(inNode);
                        mark_dirty_by_dependNodes(inNode, bOn, nodesRange, inParam);
                    }
                }
            }
        }

        if (auto pSubnetNode = dynamic_cast<SubnetNode*>(spCurrNode))
        {
            auto subinputMarkDirty = [&pSubnetNode, &nodesRange](bool dirty, std::string paramName) {
                if (auto subinput = pSubnetNode->get_subgraph()->getNode(paramName))
                    mark_dirty_by_dependNodes(subinput, dirty, nodesRange);
            };
            if (inParamName != "") {
                subinputMarkDirty(bOn, inParamName);
            }
            else {
                auto pImpl = pSubnetNode;
                for (auto& param : pImpl->get_input_primitive_params())
                    subinputMarkDirty(bOn, param.name);
                for (auto& param : pImpl->get_input_object_params())
                    subinputMarkDirty(bOn, param.name);
            }
        }

        auto spGraph = spCurrNode->getGraph();
        assert(spGraph);
        if (spGraph->getParentSubnetNode() && spCurrNode->get_nodecls() == "SubOutput")
        {
            auto parentSubgNode = spGraph->getParentSubnetNode();
            auto parentSubgNodeMarkDirty = [&nodesRange, &parentSubgNode](std::string innode, std::string inParam) {
                if (auto graph = parentSubgNode->getThisGraph()) {
                    auto inNode = graph->getNode(innode);
                    assert(inNode);
                    mark_dirty_by_dependNodes(inNode, true, nodesRange, inParam);
                }
            };
            bool find = false;
            const auto& parentSubgNodeOutputPrim = parentSubgNode->get_output_prim_param(spCurrNode->get_name(), &find);
            if (find) {
                for (auto link : parentSubgNodeOutputPrim.links) {
                    parentSubgNodeMarkDirty(link.inNode, link.inParam);
                }
            }
            bool find2 = false;
            const auto& parentSubgNodeOutputObjs = parentSubgNode->get_output_obj_param(spCurrNode->get_name(), &find2);
            if (find2) {
                for (auto link : parentSubgNodeOutputObjs.links) {
                    parentSubgNodeMarkDirty(link.inNode, link.inParam);
                }
            }
            parentSubgNode->mark_dirty(true, Dirty_All, true, false);
        }
    }

    bool isSubnetInputOutputParam(NodeImpl* spParentnode, std::string paramName)
    {
        if (SubnetNode* spSubnetnode = dynamic_cast<SubnetNode*>(spParentnode)) {
            if (auto node = spSubnetnode->get_subgraph()->getNode(paramName)) {
                if (node->get_nodecls() == "SubInput" || node->get_nodecls() == "SubOutput") {
                    return true;
                }
            }
        }
        return false;
    }

    AttrVar abiAnyToAttrVar(const zeno::reflect::Any& anyval) {
        //可以支持原有的非abi兼容的类型
        size_t code = anyval.type().hash_code();
        if (code == zeno::reflect::type_info<int>().hash_code()) {
            return zeno::reflect::any_cast<int>(anyval);
        }
        else if (code == zeno::reflect::type_info<float>().hash_code()) {
            return zeno::reflect::any_cast<float>(anyval);
        }
        else if (code == gParamType_String) {
            return zeno::any_cast_to_string(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::String>().hash_code()) {
            return zsString2Std(zeno::reflect::any_cast<zeno::String>(anyval));
        }
        else if (code == gParamType_Vec3f) {
            return zeno::reflect::any_cast<zeno::vec3f>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec3f>().hash_code()) {
            return toVec3f(zeno::reflect::any_cast<zeno::Vec3f>(anyval));
        }
        else if (code == gParamType_Vec3i) {
            return zeno::reflect::any_cast<zeno::vec3i>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec3i>().hash_code()) {
            return toVec3i(zeno::reflect::any_cast<zeno::Vec3i>(anyval));
        }
        else if (code == gParamType_Vec2i) {
            return zeno::reflect::any_cast<zeno::vec2i>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec2i>().hash_code()) {
            return toVec2i(zeno::reflect::any_cast<zeno::Vec2i>(anyval));
        }
        else if (code == gParamType_Vec2f) {
            return zeno::reflect::any_cast<zeno::vec2f>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec2f>().hash_code()) {
            return toVec2f(zeno::reflect::any_cast<zeno::Vec2f>(anyval));
        }
        else if (code == gParamType_Vec4i) {
            return zeno::reflect::any_cast<zeno::vec4i>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec4i>().hash_code()) {
            return toVec4i(zeno::reflect::any_cast<zeno::Vec4i>(anyval));
        }
        else if (code == gParamType_Vec4f) {
            return zeno::reflect::any_cast<zeno::vec4f>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vec4f>().hash_code()) {
            return toVec4f(zeno::reflect::any_cast<zeno::Vec4f>(anyval));
        }
        else if (code == gParamType_IntList) {
            return zeno::reflect::any_cast<std::vector<int>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<int>>().hash_code()) {
            return zeVec2stdVec(any_cast<zeno::Vector<int>>(anyval));
        }
        else if (code == gParamType_FloatList) {
            return zeno::reflect::any_cast<std::vector<float>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<float>>().hash_code()) {
            return zeVec2stdVec(any_cast<zeno::Vector<float>>(anyval));
        }
        else if (code == gParamType_StringList) {
            return zeno::reflect::any_cast<std::vector<std::string>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::String>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::String>>(anyval);
            std::vector<std::string> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = zsString2Std(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec3iList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec3i>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec3i>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec3i>>(anyval);
            std::vector<zeno::vec3i> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec3i(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec3fList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec3f>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec3f>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec3f>>(anyval);
            std::vector<zeno::vec3f> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec3f(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec2iList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec2i>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec2i>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec2i>>(anyval);
            std::vector<zeno::vec2i> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec2i(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec2fList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec2f>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec2f>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec2f>>(anyval);
            std::vector<zeno::vec2f> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec2f(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec4fList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec4f>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec4f>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec4f>>(anyval);
            std::vector<zeno::vec4f> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec4f(zvec[i]);
            }
            return vec;
        }
        else if (code == gParamType_Vec4iList) {
            return zeno::reflect::any_cast<std::vector<zeno::vec4i>>(anyval);
        }
        else if (code == zeno::reflect::type_info<zeno::Vector<zeno::Vec4i>>().hash_code()) {
            auto zvec = any_cast<zeno::Vector<zeno::Vec4i>>(anyval);
            std::vector<zeno::vec4i> vec(zvec.size());
            for (int i = 0; i < zvec.size(); i++) {
                vec[i] = toVec4i(zvec[i]);
            }
            return vec;
        }
        else {
            return AttrVar();
        }
    }

    void update_list_root_key(ListObject* listobj, const std::string& key)
    {
        std::string listkey = zsString2Std(listobj->key());
        auto it = listkey.find_first_of('\\');
        std::string newkey = it == std::string::npos ? listkey : (key + listkey.substr(it));
        listobj->update_key(stdString2zs(newkey));

        std::set<std::string> mmodify, mnew, mremove;
        for (const auto& uuid : listobj->m_impl->m_modify) {
            auto it = uuid.find_first_of('\\');
            mmodify.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        for (const auto& uuid : listobj->m_impl->m_new_added) {
            auto it = uuid.find_first_of('\\');
            mnew.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        for (auto& uuid : listobj->m_impl->m_new_removed) {
            auto it = uuid.find_first_of('\\');
            mremove.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        mmodify.swap(listobj->m_impl->m_modify);
        mnew.swap(listobj->m_impl->m_new_added);
        mremove.swap(listobj->m_impl->m_new_removed);
        for (auto obj : listobj->m_impl->get()) {
            if (auto plist = dynamic_cast<ListObject*>(obj)) {
                update_list_root_key(plist, key);
            } else if (auto pdict = dynamic_cast<DictObject*>(obj)) {
                update_dict_root_key(pdict, key);
            } else {
                std::string objkey = zsString2Std(obj->key());
                auto it = objkey.find_first_of('\\');
                newkey = it == std::string::npos ? objkey : (key + objkey.substr(it));
                obj->update_key(stdString2zs(newkey));
            }
        }
    }

    void add_prefix_key(IObject* pObject, const std::string& prefix) {
        zeno::String newKey = stdString2zs(prefix) + '\\' + pObject->key();
        pObject->update_key(newKey);
        if (ListObject* pList = dynamic_cast<ListObject*>(pObject)) {
            for (auto& elemObj : pList->m_impl->m_objects) {
                add_prefix_key(elemObj.get(), prefix);
            }
            std::set<std::string> modify, removed, added;
            for (auto id : pList->m_impl->m_new_added) {
                added.insert(prefix + '\\' + id);
            }
            for (auto id : pList->m_impl->m_modify) {
                modify.insert(prefix + '\\' + id);
            }
            for (auto id : pList->m_impl->m_new_removed) {
                removed.insert(prefix + '\\' + id);
            }
            pList->m_impl->m_new_added = added;
            pList->m_impl->m_modify = modify;
            pList->m_impl->m_new_removed = removed;
        }
        else if (DictObject* pDict = dynamic_cast<DictObject*>(pObject)) {
            for (auto& [key, spObject] : pDict->lut) {
                add_prefix_key(spObject.get(), prefix);
            }
        }
    }

    zany clone_by_key(IObject* pObject, const std::string& prefix) {
        if (ListObject* pList = dynamic_cast<ListObject*>(pObject)) {
            auto newList = create_ListObject();
            zeno::String newKey = stdString2zs(prefix) + '\\' + pList->key();
            newList->update_key(newKey);
            for (const auto& uuid : pList->m_impl->m_modify) {
                std::string _key = prefix + '\\' + uuid;
                newList->m_impl->m_modify.insert(_key);
            }
            for (const auto& uuid : pList->m_impl->m_new_added) {
                std::string _key = prefix + '\\' + uuid;
                newList->m_impl->m_new_added.insert(_key);
            }
            for (const auto& uuid : pList->m_impl->m_new_removed) {
                std::string _key = prefix + '\\' + uuid;
                newList->m_impl->m_new_removed.insert(_key);
            }

            for (auto& spObject : pList->m_impl->m_objects) {
                auto newObj = clone_by_key(spObject.get(), prefix);
                newList->m_impl->m_objects.push_back(std::move(newObj));
            }
            return newList;
        }
        else if (DictObject* pDict = dynamic_cast<DictObject*>(pObject)) {
            //不修改自身
            auto newDict = create_DictObject();
            //DictObject* newDict = create_DictObject();
            newDict->update_key(stdString2zs(prefix) + '\\' + pDict->key());

            for (const auto& uuid : pDict->m_modify) {
                newDict->m_modify.insert(prefix + '\\' + uuid);
            }
            for (const auto& uuid : pDict->m_new_added) {
                newDict->m_new_added.insert(prefix + '\\' + uuid);
            }
            for (const auto& uuid : pDict->m_new_removed) {
                newDict->m_new_removed.insert(prefix + '\\' + uuid);
            }
            for (auto& [key, spObject] : pDict->lut) {
                std::string new_key = prefix + '\\' + key;
                auto newObj = clone_by_key(spObject.get(), prefix);
                newDict->lut.insert(std::make_pair(key, std::move(newObj)));
            }
            return newDict;
        }
        else {
            auto spClonedObj = pObject->clone();
            zeno::String newkey = stdString2zs(prefix) + '\\' + pObject->key();
            spClonedObj->update_key(newkey);
            return spClonedObj;
        }
    }

    std::vector<std::string> get_obj_paths(IObject* pObject) {
        std::vector<std::string> _paths;
        if (ListObject* lstObj = dynamic_cast<ListObject*>(pObject)) {
            for (auto spObj : lstObj->get()) {
                std::vector<std::string> subpaths = get_obj_paths(spObj);
                for (const std::string& subpath : subpaths) {
                    _paths.push_back(subpath);
                }
            }
            return _paths;
        }
        else {
            _paths.push_back(zsString2Std(pObject->key()));
            return _paths;
        }
    }

    void update_dict_root_key(DictObject* dictobj, const std::string& key)
    {
        std::string dictkey = zsString2Std(dictobj->key());
        auto it = dictkey.find_first_of('\\');
        std::string newkey = it == std::string::npos ? key : (key + dictkey.substr(it));
        dictobj->update_key(stdString2zs(newkey));

        std::set<std::string> mmodify, mnew, mremove;
        for (const auto& uuid : dictobj->m_modify) {
            auto it = uuid.find_first_of('\\');
            mmodify.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        for (const auto& uuid : dictobj->m_new_added) {
            auto it = uuid.find_first_of('\\');
            mnew.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        for (auto& uuid : dictobj->m_new_removed) {
            auto it = uuid.find_first_of('\\');
            mremove.insert(it == std::string::npos ? uuid : key + uuid.substr(it));
        }
        mmodify.swap(dictobj->m_modify);
        mnew.swap(dictobj->m_new_added);
        mremove.swap(dictobj->m_new_removed);
        for (auto& [str, obj] : dictobj->lut) {
            if (auto plist = dynamic_cast<ListObject*>(obj.get())) {
                update_list_root_key(plist, key);
            } else if (auto pdict = dynamic_cast<DictObject*>(obj.get())) {
                update_dict_root_key(pdict, key);
            } else {
                std::string objkey = zsString2Std(obj->key());
                auto it = objkey.find_first_of('\\');
                newkey = it == std::string::npos ? objkey : (key + objkey.substr(it));
                obj->update_key(stdString2zs(newkey));
            }
        }
    }

    bool getParamInfo(const CustomUI& customui, std::vector<ParamPrimitive>& inputs, std::vector<ParamPrimitive>& outputs) {
        return false;
    }

    zeno::ParamType findParamType(const CustomUI& customui, bool bInput, const std::string& name) {
        if (bInput) {
            for (auto param : customui.inputObjs) {
                if (param.name == name) {
                    return param.type;
                }
            }
            for (auto tab : customui.inputPrims) {
                for (auto group : tab.groups) {
                    for (auto param : group.params) {
                        if (param.name == name) {
                            return param.type;
                        }
                    }
                }
            }
        }
        else {
            for (auto param : customui.outputPrims) {
                if (param.name == name) {
                    return param.type;
                }
            }
            for (auto param : customui.outputObjs) {
                if (param.name == name) {
                    return param.type;
                }
            }
        }
        return Param_Null;
    }

    bool isPrimitiveType(const ParamType type) {
        //这个是给旧式定义节点使用的，新的反射定义方式不再使用，其初始化过程也不会走到这里判断。
        return type == gParamType_String ||
            type == gParamType_Int ||
            type == gParamType_Float ||
            type == gParamType_Vec2i ||
            type == gParamType_Vec3i ||
            type == gParamType_Vec4i ||
            type == gParamType_Vec2f ||
            type == gParamType_Vec3f ||
            type == gParamType_Vec4f ||
            type == gParamType_AnyNumeric ||
            type == gParamType_Bool ||
            type == gParamType_Heatmap ||
            type == gParamType_Curve ||
            type == gParamType_Shader ||
            type == gParamType_StringList ||
            type == gParamType_Matrix4 ||
            type == gParamType_ListOfMat4 ||
            type == gParamType_IntList ||
            type == gParamType_FloatList ||
            type == gParamType_GLMVec3;
    }

    zany strToZAny(std::string const& defl, ParamType const& type) {
        switch (type) {
        case gParamType_String: {
            zany res = std::make_unique<zeno::StringObject>(defl);
            return res;
        }
        case gParamType_Int: {
            return std::make_unique<NumericObject>(std::stoi(defl));
        }
        case gParamType_Float: {
            return std::make_unique<NumericObject>(std::stof(defl));
        }
        case gParamType_Vec2i:
        case gParamType_Vec3i:
        case gParamType_Vec4i:
        {
            std::vector<int> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stoi(v));
            }

            if (gParamType_Vec2i == type) {
                return std::make_unique<NumericObject>(vec2i(vec[0], vec[1]));
            }
            else if (gParamType_Vec3i == type) {
                return std::make_unique<NumericObject>(vec3i(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_unique<NumericObject>(vec4i(vec[0], vec[1], vec[2], vec[3]));
            }
        }
        case gParamType_Vec2f:
        case gParamType_Vec3f:
        case gParamType_Vec4f:
        {
            std::vector<float> vec;
            for (auto v : split_str(defl, ',')) {
                vec.push_back(std::stof(v));
            }

            if (gParamType_Vec2f == type) {
                return std::make_unique<NumericObject>(vec2f(vec[0], vec[1]));
            }
            else if (gParamType_Vec3f == type) {
                return std::make_unique<NumericObject>(vec3f(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_unique<NumericObject>(vec4f(vec[0], vec[1], vec[2], vec[3]));
            }
        }
        default:
            return nullptr;
        }
    }

    bool isEqual(const zvariant& lhs, const zvariant& rhs, ParamType const type) {
        if (lhs.index() != rhs.index())
            return false;

        std::visit([&](auto&& arg1, auto&& arg2) -> bool {
            using T = std::decay_t<decltype(arg1)>;
            using E = std::decay_t<decltype(arg2)>;
            if constexpr (std::is_same_v<T, int> && std::is_same_v<E, int>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, float> && std::is_same_v<E, float>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, std::string> && std::is_same_v<E, std::string>) {
                return arg1 == arg2;
            }
            else if constexpr (std::is_same_v<T, zeno::vec2i> && std::is_same_v<E, zeno::vec2i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec2f> && std::is_same_v<E, zeno::vec2f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec3i> && std::is_same_v<E, zeno::vec3i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg2[2]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec3f> && std::is_same_v<E, zeno::vec3f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg2[2]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec4i> && std::is_same_v<E, zeno::vec4i>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg2[2] && arg1[3] == arg2[3]);
            }
            else if constexpr (std::is_same_v<T, zeno::vec4f> && std::is_same_v<E, zeno::vec4f>)
            {
                return (arg1[0] == arg2[0] && arg1[1] == arg2[1] && arg1[2] == arg2[2] && arg1[3] == arg2[3]);
            }
            else
            {
                return false;
            }
            }, lhs, rhs);
    }

    ZENO_API std::string editVariantToStr(const PrimVar& var)
    {
        return std::visit([](auto&& val) -> std::string {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, int>) {
                return std::to_string(val);
            }
            else if constexpr (std::is_same_v<T, float>) {
                return std::to_string(val);
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                return val;
            }
            else {
                return "";
            }
            }, var);
    }

    ZENO_API zeno::ParamControl getDefaultControl(const zeno::ParamType type)
    {
        switch (type)
        {
        case Param_Null:      return zeno::NullControl;
        case gParamType_Bool:      return zeno::Checkbox;
        case gParamType_Int:       return zeno::Lineedit;
        case gParamType_String:    return zeno::Lineedit;
        case gParamType_Float:     return zeno::Lineedit;
        case gParamType_Vec2i:     return zeno::Vec2edit;
        case gParamType_Vec3i:     return zeno::Vec3edit;
        case gParamType_Vec4i:     return zeno::Vec4edit;
        case gParamType_Vec2f:     return zeno::Vec2edit;
        case gParamType_Vec3f:     return zeno::Vec3edit;
        case gParamType_Vec4f:     return zeno::Vec4edit;
        case gParamType_Primitive:
        case gParamType_Dict:
        case gParamType_List:      return zeno::NullControl;
            //Param_Color:  //need this?
        case gParamType_Curve:     return zeno::CurveEditor;
        case gParamType_Heatmap: return zeno::Heatmap;
        default:
            return zeno::NullControl;
        }
    }

    ZENO_API std::string getControlDesc(zeno::ParamControl ctrl, zeno::ParamType type)
    {
        switch (ctrl)
        {
        case zeno::Lineedit:
        {
            switch (type) {
            case gParamType_Float:     return "Float";
            case gParamType_Int:       return "Integer";
            case gParamType_String:    return "String";
            }
            return "";
        }
        case zeno::Checkbox:
        {
            return "Boolean";
        }
        case zeno::Multiline:           return "Multiline String";
        case zeno::ReadPathEdit:            return "read path";
        case zeno::WritePathEdit:            return "write path";
        case zeno::DirectoryPathEdit:            return "directory";
        case zeno::Combobox:            return "Enum";
        case zeno::Vec4edit:
        {
            return type == gParamType_Int ? "Integer Vector 4" : "Float Vector 4";
        }
        case zeno::Vec3edit:
        {
            return type == gParamType_Int ? "Integer Vector 3" : "Float Vector 3";
        }
        case zeno::Vec2edit:
        {
            return type == gParamType_Int ? "Integer Vector 2" : "Float Vector 2";
        }
        case zeno::Heatmap:             return "Color";
        case zeno::ColorVec:            return "Color Vec3f";
        case zeno::CurveEditor:         return "Curve";
        case zeno::SpinBox:             return "SpinBox";
        case zeno::DoubleSpinBox:       return "DoubleSpinBox";
        case zeno::Slider:              return "Slider";
        case zeno::SpinBoxSlider:       return "SpinBoxSlider";
        case zeno::Seperator:           return "group-line";
        case zeno::PythonEditor:        return "PythonEditor";
        case zeno::CodeEditor:          return "CodeEditor";
        default:
            return "";
        }
    }
}