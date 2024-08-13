#include <zeno/utils/helper.h>
#include <regex>
#include <zeno/core/CoreParam.h>
#include <zeno/core/INode.h>
#include <zeno/types/ObjectDef.h>
#include <regex>


using namespace zeno::types;
using namespace zeno::reflect;

namespace zeno {

    ZENO_API ParamType convertToType(std::string const& type, const std::string_view& param_name) {
        //TODO: deprecated literal representation.
        if (type == "string" || type == "readpath" || type == "writepath" || type == "diratory" || type == "multiline_string")
        { return gParamType_String; }
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
        else if (type == "prim" || type == "PrimitiveObject" || type == "primitive") { return gParamType_Primitive; }
        else if (type == "list" || type == "ListObject") { return gParamType_List; }
        else if (type == "dict" || type == "DictObject" || type == "DictObject:NumericObject") { return gParamType_Dict; }
        else if (type == "colorvec3f") { return gParamType_Vec3f; }
        else if (type == "color") { return gParamType_Heatmap; }
        else if (type == "curve") { return gParamType_Curve; }
        else if (starts_with(type, "enum ")) { return gParamType_String; }
        else if (type == "AxisObject") { return gParamType_sharedIObject; }
        else if (type == "CameraObject") { return gParamType_Camera; }
        else if (type == "LightObject") { return gParamType_Light; }
        else if (type == "FunctionObject") { return gParamType_sharedIObject; }
        else if (type == "object" ||
                type == "IObject" || 
                type == "zany" || 
                type == "material" ||
                type == "texture" ||
                type == "instancing" ||
                type == "shader" ||
                type == "MaterialObject" ||
                type == "LBvh") {
            return gParamType_sharedIObject; 
    }
        else if (type == "VDBGrid") {
            return gParamType_sharedIObject;
        }
        else if (type == ""){
            //类型名字为空时，只能根据参数名字去猜测
            if (param_name == "prim") {
                return gParamType_Primitive;
            }
            else if (param_name == "object") {
                return gParamType_sharedIObject;
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
                return gParamType_sharedIObject;    //只能给Object了，不然就要再分配一个枚举值
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
                return gParamType_sharedIObject;
            }
            else if (param_name == "VDBGrid" || param_name == "grid") {
                return gParamType_sharedIObject;
            }
            else if (param_name == "heatmap") {
                return gParamType_Heatmap;
            }
            else {
                return gParamType_sharedIObject;
            }
        }
        else
            return gParamType_sharedIObject;    //zeno各个模块定义的类型不规范程度很大，而且积累了很多，很难一下子改好，所以不明类型都转成obj
    }

    ZENO_API bool isAnyEqual(const zeno::reflect::Any& lhs, const zeno::reflect::Any& rhs)
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
        case gParamType_Float:  return zeno::reflect::any_cast<float>(lhs) == zeno::reflect::any_cast<float>(rhs);
        case gParamType_String: return zeno::reflect::any_cast<std::string>(lhs) == zeno::reflect::any_cast<std::string>(rhs);
        case gParamType_Vec2f: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec2f>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec2f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec2i: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec2i>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec2i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec2s: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec2s>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec2s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1];
        }
        case gParamType_Vec3f: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec3f>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec3f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec3i: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec3i>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec3i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec3s: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec3s>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec3s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2];
        }
        case gParamType_Vec4f: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec4f>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec4f>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Vec4i: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec4i>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec4i>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Vec4s: {
            auto& vec1 = zeno::reflect::any_cast<zeno::vec4s>(lhs);
            auto& vec2 = zeno::reflect::any_cast<zeno::vec4s>(rhs);
            return vec1[0] == vec2[0] && vec1[1] == vec2[1] && vec1[2] == vec2[2] && vec1[3] == vec2[3];
        }
        case gParamType_Curve: {
            auto& curve1 = zeno::reflect::any_cast<zeno::CurvesData>(lhs);
            auto& curve2 = zeno::reflect::any_cast<zeno::CurvesData>(rhs);
            return curve1 == curve2;
        }
        default:
            return false;
        }
    }

    ZENO_API std::string paramTypeToString(ParamType type)
    {
        switch (type)
        {
        case Param_Null:    return "";
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
        case gParamType_Primitive:    return "prim";
        case gParamType_Dict:    return "dict";
        case gParamType_List:    return "list";
        case gParamType_Curve:   return "curve";
        case gParamType_Heatmap: return "color";
        default:
            return "";
        }
    }

    ZENO_API zeno::reflect::Any str2any(std::string const& defl, ParamType const& type) {
        if (defl.empty())
            return initAnyDeflValue(type);
        switch (type) {
        case gParamType_String: {
            return defl;
        }
        case gParamType_Bool: {
            if (defl == "0" || defl == "false")    return 0;
            if (defl == "1" || defl == "true")     return 1;
            return zeno::reflect::Any();
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
            return zeno::reflect::Any();
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
            return zeno::reflect::Any();
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

    ZENO_API zvariant AnyToZVariant(zeno::reflect::Any const& var) {
        if (!var.has_value())
            return zvariant();
        if (zeno::reflect::get_type<int>() == var.type()) {
            return zeno::reflect::any_cast<int>(var);
        }
        else if (zeno::reflect::get_type<float>() == var.type()) {
            return zeno::reflect::any_cast<float>(var);
        }
        else if (zeno::reflect::get_type<std::string>() == var.type()) {
            return zeno::reflect::any_cast<std::string>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec2i>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec2i>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec3i>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec3i>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec4i>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec4i>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec2f>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec2f>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec3f>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec3f>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec4f>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec4f>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec2s>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec2s>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec3s>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec3s>(var);
        }
        else if (zeno::reflect::get_type<zeno::vec4s>() == var.type()) {
            return zeno::reflect::any_cast<zeno::vec4s>(var);
        }
        return zvariant();
    }

    ZENO_API zeno::reflect::Any initAnyDeflValue(ParamType const& type)
    {
        if (type == gParamType_String) {
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
            return (int)0;
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
            return zeno::reflect::make_any<CurvesData>();
        }
        else if (type == gParamType_sharedIObject)
        {
            return std::shared_ptr<IObject>();
        }
        return zeno::reflect::Any();
    }

    zvariant zeno::initDeflValue(ParamType const& type)
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

        auto spOutNode = spOutParam->m_wpNode.lock();
        auto spInNode = spInParam->m_wpNode.lock();
        if (!spOutNode || !spInNode)
            return edge;

        const std::string& outNode = spOutNode->get_name();
        const std::string& outParam = spOutParam->name;
        const std::string& inNode = spInNode->get_name();
        const std::string& inParam = spInParam->name;
        edge = { outNode, outParam, spLink->fromkey, inNode, inParam, spLink->tokey, true };
        return edge;
    }

    EdgeInfo getEdgeInfo(std::shared_ptr<PrimitiveLink> spLink) {
        EdgeInfo edge;
        auto spOutParam = spLink->fromparam;
        auto spInParam = spLink->toparam;
        if (!spOutParam || !spInParam)
            return edge;

        auto spOutNode = spOutParam->m_wpNode.lock();
        auto spInNode = spInParam->m_wpNode.lock();
        if (!spOutNode || !spInNode)
            return edge;

        const std::string& outNode = spOutNode->get_name();
        const std::string& outParam = spOutParam->name;
        const std::string& inNode = spInNode->get_name();
        const std::string& inParam = spInParam->name;
        edge = { outNode, outParam, "", inNode, inParam, "", false};
        return edge;
    }

    std::string generateObjKey(std::shared_ptr<IObject> spObject) {
        return "";    //TODO
    }

    ZENO_API std::string objPathToStr(ObjPath path) {
        return path;
    }

    ObjPath zeno::strToObjPath(const std::string& str)
    {
        return str;
    }

    PrimitiveParams customUiToParams(const CustomUIParams& customparams) {
        PrimitiveParams params;
        for (auto tab : customparams.tabs) {
            for (auto group : tab.groups) {
                params.insert(params.end(), group.params.begin(), group.params.end());
            }
        }
        return params;
    }

    ZENO_API void parseUpdateInfo(const CustomUI& customui, ParamsUpdateInfo& infos)
    {
        for (const zeno::ParamTab& tab : customui.inputPrims.tabs)
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

    static zeno::reflect::RTTITypeInfo getRttiInfo(ParamType type)
    {
        switch (type)
        {
        case gParamType_Bool:    return zeno::reflect::type_info<bool>();
        case gParamType_Int:     return zeno::reflect::type_info<int>();
        case gParamType_String:  return zeno::reflect::type_info<std::string>();
        case gParamType_Float:   return zeno::reflect::type_info<float>();
        case gParamType_Vec2i:   return zeno::reflect::type_info<zeno::vec2i>();
        case gParamType_Vec3i:   return zeno::reflect::type_info<zeno::vec3i>();
        case gParamType_Vec4i:   return zeno::reflect::type_info<zeno::vec4i>();
        case gParamType_Vec2f:   return zeno::reflect::type_info<zeno::vec2f>();
        case gParamType_Vec3f:   return zeno::reflect::type_info<zeno::vec3f>();
        case gParamType_Vec4f:   return zeno::reflect::type_info<zeno::vec4f>();
        case gParamType_sharedIObject://TODO: vdbgrid: VDBFloatGrid or VDBFloat3Grid VDBPointsGrid?
            return zeno::reflect::type_info<std::shared_ptr<IObject>>();
        case gParamType_Primitive:    return zeno::reflect::type_info<std::shared_ptr<zeno::PrimitiveObject>>();
        case gParamType_Camera:  return zeno::reflect::type_info<std::shared_ptr<zeno::CameraObject>>();
        case gParamType_Light:   return zeno::reflect::type_info<std::shared_ptr<zeno::LightObject>>();
        case gParamType_Dict:    return zeno::reflect::type_info<std::shared_ptr<zeno::DictObject>>();
        case gParamType_List:    return zeno::reflect::type_info<std::shared_ptr<zeno::ListObject>>();
        case gParamType_Curve:   return zeno::reflect::type_info<zeno::CurvesData>();
        case gParamType_Heatmap: return zeno::reflect::type_info<zeno::HeatmapData>();
        default:
            //no supporting right now.
            //assert(false);
            return { "<default_type>", 0, 0 };
        }
    }

    void initControlsByType(CustomUI& ui) {
        for (ParamTab& tab : ui.inputPrims.tabs)
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

    std::set<std::string> zeno::getReferPaths(const zvariant& val)
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

    std::string zeno::relativePath(std::string currentPath, const std::string& path)
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

    bool isObjectType(const zeno::reflect::RTTITypeInfo& type, bool& isConstPtr)
    {
        isConstPtr = false;
        std::string name(type.name());
        //目前没有太多办法很方便判断是否为对象类型，如果采用枚举值，就得不断登记，先用这种方式，后续可能采用register的方式。
        if (name.find("shared_ptr<") != std::string::npos) {
            isConstPtr = name.find("shared_ptr<const ") != std::string::npos;
            return true;
        }
        else {
            return false;
        }
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
        if (type == types::gParamType_Int || type == types::gParamType_Float)
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
        if (left == types::gParamType_Vec2i && right == types::gParamType_Vec2f || left == types::gParamType_Vec2f && right == types::gParamType_Vec2i ||
            left == types::gParamType_Vec3i && right == types::gParamType_Vec3f || left == types::gParamType_Vec3f && right == types::gParamType_Vec3i ||
            left == types::gParamType_Vec4i && right == types::gParamType_Vec4f || left == types::gParamType_Vec4f && right == types::gParamType_Vec4i)
            return true;
        return false;
    }

    ZENO_API bool outParamTypeCanConvertInParamType(ParamType outType, ParamType inType)
    {
        if (isNumericType(outType) && isNumericVecType(inType)) {   //数值连数值vec
            return true;
        }
        else if (isSameDimensionNumericVecType(outType, inType)) { //同维度数值vec互连
            return true;
        }
        else if (zeno::reflect::get_type<std::shared_ptr<zeno::IObject>>().type_hash() == inType && isObjectType(outType)) {    //outType的Obj类型可以转IObject
            return true;
        }
        else{
            return false;
        }
    }

    bool getParamInfo(const CustomUI& customui, std::vector<ParamPrimitive>& inputs, std::vector<ParamPrimitive>& outputs) {
        return false;
    }

    bool isPrimitiveType(const ParamType type) {
        //这个是给旧式定义节点使用的，新的反射定义方式不再使用，其初始化过程也不会走到这里判断。
        return type == gParamType_String || type == gParamType_Int || type == gParamType_Float || type == gParamType_Vec2i ||
            type == gParamType_Vec3i || type == gParamType_Vec4i || type == gParamType_Vec2f || type == gParamType_Vec3f ||
            type == gParamType_Vec4f || type == gParamType_Bool || type == gParamType_Heatmap || type == gParamType_Curve;//TODO: heatmap type.
    }

    zany strToZAny(std::string const& defl, ParamType const& type) {
        switch (type) {
        case gParamType_String: {
            zany res = std::make_shared<zeno::StringObject>(defl);
            return res;
        }
        case gParamType_Int: {
            return std::make_shared<NumericObject>(std::stoi(defl));
        }
        case gParamType_Float: {
            return std::make_shared<NumericObject>(std::stof(defl));
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
                return std::make_shared<NumericObject>(vec2i(vec[0], vec[1]));
            }
            else if (gParamType_Vec3i == type) {
                return std::make_shared<NumericObject>(vec3i(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_shared<NumericObject>(vec4i(vec[0], vec[1], vec[2], vec[3]));
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
                return std::make_shared<NumericObject>(vec2f(vec[0], vec[1]));
            }
            else if (gParamType_Vec3f == type) {
                return std::make_shared<NumericObject>(vec3f(vec[0], vec[1], vec[2]));
            }
            else {
                return std::make_shared<NumericObject>(vec4f(vec[0], vec[1], vec[2], vec[3]));
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
        case zeno::CodeEditor:        return "CodeEditor";
        default:
            return "";
        }
    }
}