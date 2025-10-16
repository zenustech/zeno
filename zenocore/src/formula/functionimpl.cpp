#include "functionimpl.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/Error.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/utils/helper.h>
#include <regex>
#include <variant>
#include <functional>
#include <zeno/utils/format.h>
#include <numeric>
#include <zeno/geo/geometryutil.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/vectorutil.h>
#include <zeno/utils/interfaceutil.h>
#include <zeno/core/data.h>
#include <zeno/core/FunctionManager.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/extra/CalcContext.h>
#include <zeno/geo/kdsearch.h>
#include <random>
#include "../utils/zfxutil.h"

#define REF_DEPEND_APPLY

namespace zeno
{
    namespace zfx
    {
        template <class T>
        static T get_zfxvar(zfxvariant value) {
            return std::visit([](auto const& val) -> T {
                using V = std::decay_t<decltype(val)>;
                if constexpr (!std::is_constructible_v<T, V>) {
                    if constexpr (std::is_same_v<T, glm::vec3> && (std::is_same_v<V, zfxfloatarr> || std::is_same_v<V, zfxintarr>)) {
                        return glm::vec3(val[0], val[1], val[2]);
                    } else if constexpr (std::is_same_v<T, zeno::vec3f> && (std::is_same_v<V, zfxfloatarr> || std::is_same_v<V, zfxintarr>)) {
                        return zeno::vec3f(val[0], val[1], val[2]);
                    } else if constexpr (std::is_same_v<T, zeno::vec3i> && (std::is_same_v<V, zfxfloatarr> || std::is_same_v<V, zfxintarr>)) {
                        return zeno::vec3i(val[0], val[1], val[2]);
                    } 
                    throw makeError<TypeError>(typeid(T), typeid(V), "get<zfxvariant>");
                }
                else {
                    return T(val);
                }
                }, value);
        }

        template <class T>
        static T get_zfxvec_front_elem(const ZfxVector& value) {
            return std::visit([](auto const& vec) -> T {
                using V = std::decay_t<decltype(vec)>;
                using E = typename V::value_type;
                const auto& val = vec[0];
                if constexpr (!std::is_constructible_v<T, E>) {
                    if constexpr (std::is_same_v<T, glm::vec3> && (std::is_same_v<E, zfxfloatarr> || std::is_same_v<E, zfxintarr>)) {
                        return glm::vec3(val[0], val[1], val[2]);
                    }
                    else if constexpr (std::is_same_v<T, zeno::vec3f> && (std::is_same_v<E, zfxfloatarr> || std::is_same_v<E, zfxintarr>)) {
                        return zeno::vec3f(val[0], val[1], val[2]);
                    }
                    else if constexpr (std::is_same_v<T, zeno::vec3i> && (std::is_same_v<E, zfxfloatarr> || std::is_same_v<E, zfxintarr>)) {
                        return zeno::vec3i(val[0], val[1], val[2]);
                    }
                    throw makeError<TypeError>(typeid(T), typeid(E), "get<zfxvariant>");
                }
                else {
                    return T(val);
                }
                }, value);
        }

        template <class T>
        static T prim_reduce(std::vector<T> const& temp, std::string type)
        {
            if (type == std::string("avg")) {
                T total = zeno::parallel_reduce_array<T>(temp.size(), T(0), [&](size_t i) -> T { return temp[i]; },
                    [&](T i, T j) -> T { return i + j; });
                return total / (T)(temp.size());
            }
            if (type == std::string("max")) {
                T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return temp[i]; },
                    [&](T i, T j) -> T { return zeno::max(i, j); });
                return total;
            }
            if (type == std::string("min")) {
                T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return temp[i]; },
                    [&](T i, T j) -> T { return zeno::min(i, j); });
                return total;
            }
            if (type == std::string("absmax")) {
                T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return zeno::abs(temp[i]); },
                    [&](T i, T j) -> T { return zeno::max(i, j); });
                return total;
            }
            return T(0);
        }

        template <class T>
        void removeElemsByIndice(std::vector<T>& vec, std::set<int> sorted_indice) {
            if (sorted_indice.empty())
                return;

            for (auto iter = sorted_indice.rbegin(); iter != sorted_indice.rend(); iter++) {
                int rmIdx = *iter;
                vec.erase(vec.begin() + rmIdx);
            }
        }

        static std::string format_variable_size(const char* fmt, std::vector<zfxvariant> args) {
            return std::accumulate(
                std::begin(args),
                std::end(args),
                std::string{ fmt },
                [](std::string toFmt, zfxvariant arg) {
                    return std::visit([toFmt](auto&& val)->std::string {
                        using T = std::decay_t<decltype(val)>;
                        if constexpr (std::is_same_v<T, int>) {
                            return format(toFmt, val);
                        }
                        else if constexpr (std::is_same_v<T, float>) {
                            return format(toFmt, val);
                        }
                        else if constexpr (std::is_same_v<T, std::string>) {
                            return format(toFmt, val);
                        }
                        else if constexpr (std::is_same_v<T, glm::vec2>) {
                            return format(toFmt, "{" + std::to_string(val.x) + "," + std::to_string(val.y) + "}");
                        }
                        else if constexpr (std::is_same_v<T, glm::vec3>) {
                            return format(toFmt, "{" + std::to_string(val.x) + "," + std::to_string(val.y )+ "," + std::to_string(val.z) + "}");
                        }
                        else if constexpr (std::is_same_v<T, glm::vec4>) {
                            return format(toFmt, "{" + std::to_string(val.x) + "," + std::to_string(val.y) + "," + std::to_string(val.z) + "," + std::to_string(val.w) + "}");
                        }
                        else {
                            throw makeError<UnimplError>("error type on format string");
                        }
                        }, arg);
                }
            );
        }

        static AttrVar zfxVarToAttrVar(const ZfxVariable& var) {
            return std::visit([&](auto& vec)->AttrVar {
                using T = std::decay_t<decltype(vec)>;
                using E = typename T::value_type;

                if constexpr (
                    std::is_same_v<E, int> ||
                    std::is_same_v<E, float> ||
                    std::is_same_v<E, std::string>)
                {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    return vec;
                }
                else if constexpr (
                    std::is_same_v<E, glm::vec2> ||
                    std::is_same_v<E, glm::vec3> ||
                    std::is_same_v<E, glm::vec4>)
                {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    return vec;
                }
                else if constexpr (
                    std::is_same_v<E, zfxintarr> ||
                    std::is_same_v<E, zfxfloatarr> ||
                    std::is_same_v<E, zfxstringarr>) {
                    throw makeError<UnimplError>("no support array as the attribute type.");
                }
                else {
                    throw makeError<UnimplError>("no supported attribute type.");
                }

                }, var.value);
        }

        static ZfxVector getAttrs(GeometryObject* spGeo, GeoAttrGroup grp, std::string name) {
            if (name == "P") name = "pos";
            if (name == "N") name = "nrm";
            
            GeoAttrType type = spGeo->get_attr_type(grp, name);
            if (type == ATTR_INT) {
                return spGeo->get_attrs<int>(grp, name);
            }
            else if (type == ATTR_FLOAT) {
                return spGeo->get_attrs<float>(grp, name);
            }
            else if (type == ATTR_STRING) {
                return spGeo->get_attrs<std::string>(grp, name);
            }
            else if (type == ATTR_VEC2) {
                const std::vector<zeno::vec2f>& zvec = spGeo->get_attrs<zeno::vec2f>(grp, name);
                std::vector<glm::vec2> vec(zvec.size());
                for (size_t i = 0; i < zvec.size(); ++i) {
                    vec[i] = glm::vec2(zvec[i][0], zvec[i][1]);
                }
                return vec;
            }
            else if (type == ATTR_VEC3) {
                const std::vector<zeno::vec3f>& zvec = spGeo->get_attrs<zeno::vec3f>(grp, name);
                std::vector<glm::vec3> vec(zvec.size());
                for (size_t i = 0; i < zvec.size(); ++i) {
                    vec[i] = glm::vec3(zvec[i][0], zvec[i][1], zvec[i][2]);
                }
                return vec;
            }
            else if (type == ATTR_VEC4) {
                const std::vector<zeno::vec4f>& zvec = spGeo->get_attrs<zeno::vec4f>(grp, name);
                std::vector<glm::vec4> vec(zvec.size());
                for (size_t i = 0; i < zvec.size(); ++i) {
                    vec[i] = glm::vec4(zvec[i][0], zvec[i][1], zvec[i][2], zvec[i][3]);
                }
                return vec;
            }
            else {
                throw makeError<UnimplError>("unknown attr on geom");
            }
        }

        static ZfxVariable getParamValueFromRef(const std::string& ref, ZfxContext* pContext) {
            ParamPrimitive paramData;
            std::vector<std::string> items;

            if (zeno::starts_with(ref, "../")) {
                //直接取上层子图节点的参数
                std::string parampath = ref.substr(3);
                items = split_str(parampath, '.');
                std::string paramname = items[0];
                NodeImpl* parSbnNode = pContext->spNode->getGraph()->getParentSubnetNode();
                if (!parSbnNode) {
                    throw makeError<UnimplError>("cannot locate parent subnetnode, when refering params");
                }
                bool bExisted = false;
                paramData = parSbnNode->get_input_prim_param(paramname, &bExisted);
                if (!bExisted) {
                    throw makeError<UnimplError>("there is no param `" + paramname + "` in subnetnode");
                }
            }
            else {
                std::string parampath, _;
                auto spNode = zfx::getNodeAndParamFromRefString(ref, pContext, _, parampath);
                items = split_str(parampath, '.');
                std::string paramname = items[0];

                bool bExist = false;
                paramData = spNode->get_input_prim_param(paramname, &bExist);
                if (!bExist) {
                    //试一下拿prim_output，有些情况，比如获取SubInput的port，是需要拿Output的
                    paramData = spNode->get_output_prim_param(paramname, &bExist);
                    if (!bExist) {
                        throw makeError<UnimplError>("the refer param doesn't exist, may be deleted before");
                    }
                }
            }

            //直接拿引用源的计算结果，所以本节点在执行前，引用源必须先执行（比如引用源的数值也是依赖另一个节点），参考代码INode::requireInput
            //，所以要在preApply的基础上作提前依赖计算

            //如果取的是“静态值”，则不要求引用源必须执行，此时要拿的数据应该是defl.

            //以上方案取决于产品设计。

            //NOTE in 2025/3/25: 应该拿计算结果而不是默认值，并且支持跨图层获取

            float res_fval = 0.f;
            if (items.size() == 1) {
#ifdef REF_DEPEND_APPLY
                if (!paramData.result.has_value()) {
                    throw makeError<UnimplError>("there is no result on refer source, should calc the source first.");
                }

                size_t primtype = paramData.result.type().hash_code();
                if (primtype == zeno::types::gParamType_Int) {
                    res_fval = zeno::reflect::any_cast<int>(paramData.result);
                }
                else if (primtype == zeno::types::gParamType_Bool) {
                    bool bret = zeno::reflect::any_cast<bool>(paramData.result);
                    ZfxVariable varres;
                    varres.value = std::vector<int>{ bret ? 1 : 0 };
                    return varres;
                }
                else if (primtype == zeno::types::gParamType_Float) {
                    res_fval = zeno::reflect::any_cast<float>(paramData.result);
                }
                else if (primtype == zeno::types::gParamType_String) {
                    auto str = zeno::reflect::any_cast<std::string>(paramData.result);
                    ZfxVariable varres;
                    varres.value = std::vector<std::string>{ str };
                    return varres;
                }
                else if (primtype == zeno::types::gParamType_Vec2f) {
                    throw makeError<UnimplError>();
                }
                else if (primtype == zeno::types::gParamType_Vec2i) {
                    throw makeError<UnimplError>();
                }
                else if (primtype == zeno::types::gParamType_Vec3f) {
                    vec3f vec = zeno::reflect::any_cast<vec3f>(paramData.result);
                    ZfxVariable varres;
                    varres.value = std::vector<glm::vec3>{ glm::vec3(vec[0], vec[1], vec[2]) };
                    return varres;
                }
                else if (primtype == zeno::types::gParamType_Vec3i) {
                    vec3i vec = zeno::reflect::any_cast<vec3i>(paramData.result);
                    ZfxVariable varres;
                    varres.value = std::vector<glm::vec3>{ glm::vec3(vec[0], vec[1], vec[2]) };
                    return varres;
                }
                else if (primtype == zeno::types::gParamType_Vec4f) {
                    throw makeError<UnimplError>();
                }
                else if (primtype == zeno::types::gParamType_Vec4i) {
                    throw makeError<UnimplError>();
                }
                else {
                    throw makeError<UnimplError>();
                }
#else
                auto refVal = paramData.defl;
                if (!refVal.has_value()) {
                    throw makeError<UnimplError>("there is no result on refer source, should calc the source first.");
                }

                size_t primtype = refVal.type().hash_code();
                if (primtype == zeno::types::gParamType_Int) {
                    res_fval = zeno::reflect::any_cast<int>(refVal);
                }
                else if (primtype == zeno::types::gParamType_Float) {
                    res_fval = zeno::reflect::any_cast<float>(refVal);
                }
                else if (primtype == zeno::types::gParamType_PrimVariant) {
                    auto refvar = zeno::reflect::any_cast<PrimVar>(refVal);
                    res_fval = std::visit([&](auto& arg)->float {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                            return (float)arg;
                        }
                        else if constexpr (std::is_same_v<T, std::string>) {
                            if (arg.find("ref(") != std::string::npos) {
                                throw makeError<UnimplError>("don't support recursive ref in this version");
                            }
                            else {
                                throw makeError<UnimplError>("don't support string right now.");
                            }
                        }
                        else {
                            throw makeError<UnimplError>("error param type from primvar");
                        }
                    }, refvar);
                }
                else {
                    throw makeError<UnimplError>();
                }
#endif
            }
            else if (items.size() == 2 &&
                (paramData.type == zeno::types::gParamType_Vec2f || paramData.type == zeno::types::gParamType_Vec2i ||
                    paramData.type == zeno::types::gParamType_Vec3f || paramData.type == zeno::types::gParamType_Vec3i ||
                    paramData.type == zeno::types::gParamType_Vec4f || paramData.type == zeno::types::gParamType_Vec4i))
            {
                if (items[1].size() != 1)
                    throw makeError<UnimplError>();

                int idx = -1;
                switch (items[1][0])
                {
                case 'x': idx = 0; break;
                case 'y': idx = 1; break;
                case 'z': idx = 2; break;
                case 'w': idx = 3; break;
                default:
                    throw makeError<UnimplError>();
                }
                if (paramData.type == zeno::types::gParamType_Vec2f || paramData.type == zeno::types::gParamType_Vec2i) {
                    if (idx < 2) {
                        res_fval = paramData.type == zeno::types::gParamType_Vec2f ? any_cast<vec2f>(paramData.result)[idx] :
                            any_cast<vec2i>(paramData.result)[idx];
                    }
                    else {
                        throw makeError<UnimplError>();
                    }
                }
                if (paramData.type == zeno::types::gParamType_Vec3f || paramData.type == zeno::types::gParamType_Vec3i) {
                    if (idx < 3) {
                        res_fval = paramData.type == zeno::types::gParamType_Vec3f ? any_cast<vec3f>(paramData.result)[idx] :
                            any_cast<vec3i>(paramData.result)[idx];
                    }
                    else {
                        throw makeError<UnimplError>();
                    }
                }
                if (paramData.type == zeno::types::gParamType_Vec4f || paramData.type == zeno::types::gParamType_Vec4i) {
                    if (idx < 4) {
                        res_fval = paramData.type == zeno::types::gParamType_Vec4f ? any_cast<vec4f>(paramData.result)[idx] :
                            any_cast<vec4i>(paramData.result)[idx];
                    }
                    else {
                        throw makeError<UnimplError>();
                    }
                }
            }
            else {
                throw makeError<UnimplError>();
            }

            ZfxVariable varres;
            varres.value = std::vector<float>{ res_fval };
            return varres;
        };

        static IObject* getObjFromRef(const std::string& ref, ZfxContext* pContext) {
            std::string parampath, _;
            auto spNode = zfx::getNodeAndParamFromRefString(ref, pContext, _, parampath);
            auto items = split_str(parampath, '.');
            std::string paramname = items[0];
            return spNode->get_output_obj(paramname);
        }

        static ZfxVariable callRef(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("only support non-attr value when using ref");

            //TODO: vec type.
            //TODO: resolve with zeno::reflect::any
            return std::visit([&](auto& vec)->ZfxVariable {
                using T = std::decay_t<decltype(vec)>;
                using E = typename T::value_type;
                if constexpr (std::is_same_v<E, std::string>) {
                    const std::string& ref = vec[0];
                    return getParamValueFromRef(ref, pContext);
                }
                else {
                    throw makeError<UnimplError>("not support type when call `callRef`");
                }
                }, args[0].value);
        }

        static ZfxVariable parameter(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1) {
                throw makeError<UnimplError>("error number of args on param(...)");
            }
            if (pContext->param_constrain.constrain_param.empty()) {
                throw makeError<UnimplError>("only support indexing param for param constrain");
            }
            return std::visit([&](auto& vec)->ZfxVariable {
                using T = std::decay_t<decltype(vec)>;
                using E = typename T::value_type;
                if constexpr (std::is_same_v<E, std::string>) {
                    const std::string& param = vec[0];

                    auto pnode = pContext->spNode;
                    ZfxLValue lval;
                    bool bExist = false;
                    lval.var = pnode->get_input_obj_param(param, &bExist);
                    if (bExist) {
                        ZfxVariable res;
                        res.value = std::vector<ZfxLValue>{ lval };
                        return res;
                    }
                    else {
                        lval.var = pnode->get_input_prim_param(param, &bExist);
                        if (bExist) {
                            ZfxVariable res;
                            res.value = std::vector<ZfxLValue>{ lval };
                            return res;
                        }
                    }
                    throw makeError<UnimplError>("the param does not exist when calling param(...)");
                }
                else {
                    throw makeError<UnimplError>("not support type when call `callRef`");
                }
                }, args[0].value);
        }

        static ZfxVariable log(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.empty()) {
                throw makeError<UnimplError>("empty args on log");
            }
            const auto& formatStr = args[0];
            assert(formatStr.size() == 1);
            std::string formatString = get_zfxvec_front_elem<std::string>(formatStr.value);

            std::vector<ZfxVariable> _args = args;
            _args.erase(_args.begin());

            //有可能是： log("format", 2, @P.x, b);  //这里的_args的元素，可能是一个或多个。
            int maxSize = 1;
            for (auto& arg : _args) {
                maxSize = max(maxSize, arg.size());
            }

            //逐个调用输出
            if (maxSize > 1) {
                //属性或相关变量的调用
                for (int i = 0; i < maxSize; i++) {
                    if (!filter[i]) continue;
                    std::vector<zfxvariant> formatargs;
                    for (int j = 0; j < _args.size(); j++) {
                        auto& arg = _args[j];
                        assert(arg.size() != 0);
                        const auto& arg_var = std::visit([&](auto& vec)->zfxvariant {
                            using T = std::decay_t<decltype(vec)>;
                            using E = typename T::value_type;
                            if (arg.size() < i) {
                                return vec[0];
                            }
                            else {
                                return vec[1];
                            }
                            }, arg.value);
                        formatargs.push_back(arg_var);
                    }
                    std::string ret = format_variable_size(formatString.c_str(), formatargs);
                    zeno::log_only_print(ret);
                }
            }
            else {
                std::vector<zfxvariant> __args;
                for (auto __arg : _args) {
                    if (__arg.size() == 0) {
                        continue;
                    }

                    const auto& arg_var = std::visit([&](auto& vec)->zfxvariant {
                        using T = std::decay_t<decltype(vec)>;
                        using E = typename T::value_type;
                        return vec[0];
                        }, __arg.value);

                    __args.push_back(arg_var);
                }
                std::string ret = format_variable_size(formatString.c_str(), __args);
                zeno::log_only_print(ret);
            }

            return ZfxVariable();
        }

        static ZfxVariable vec3(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 3)
                throw makeError<UnimplError>("the number of elements isn't 3");
            const ZfxVariable& xvar = args[0], & yvar = args[1], & zvar = args[2];
            int nx = xvar.size(), ny = yvar.size(), nz = zvar.size();
            int N = std::max(nx, std::max(ny, nz));
            ZfxVariable res;

            std::visit([&](const auto& xvec, const auto& yvec, const auto& zvec) {
                using T1 = std::decay_t<decltype(xvec)>;
                using E1 = typename T1::value_type;

                using T2 = std::decay_t<decltype(yvec)>;
                using E2 = typename T2::value_type;

                using T3 = std::decay_t<decltype(zvec)>;
                using E3 = typename T3::value_type;

                if constexpr ((std::is_same_v<E1, float> || std::is_same_v<E1, int>) &&
                    (std::is_same_v<E2, float> || std::is_same_v<E2, int>) &&
                    (std::is_same_v<E3, float> || std::is_same_v<E3, int>))
                {
                    std::vector<glm::vec3> resvec(N);
                    for (int i = 0; i < N; i++) {
                        int ni = std::min(i, nx - 1);
                        int nj = std::min(i, ny - 1);
                        int nk = std::min(i, nz - 1);
                        float x = xvec[ni];
                        float y = yvec[nj];
                        float z = zvec[nk];
                        resvec[i] = glm::vec3(x, y, z);
                    }
                    res.value = resvec;
                }
                else {
                    throw makeError<UnimplError>("incorrect type to construct vec3");
                }
                }, xvar.value, yvar.value, zvar.value);

            return res;
        }

        template <typename RESTYPE, typename F>
        ZfxVariable call_unary_numeric_func(F&& func, const ZfxVariable& arg, ZfxElemFilter& filter) {
            return std::visit([&](auto& vec)->ZfxVariable {
                using T = std::decay_t<decltype(vec)>;
                using E = typename T::value_type;

                if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                    std::vector<RESTYPE> resvec(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        if (!filter[i]) continue;
                        resvec[i] = func(vec[i]);
                    }
                    ZfxVariable res;
                    res.value = std::move(resvec);
                    return res;
                }
                else {
                    throw makeError<UnimplError>("not support type in `sin`");
                }
                }, arg.value);
        }

        static ZfxVariable sin(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const auto& arg = args[0];

            ZfxVariable res = call_unary_numeric_func<float>([](float x)->float { return std::sin(x); }, arg, filter);
            return res;
        }

        static ZfxVariable cos(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const auto& arg = args[0];
            ZfxVariable res = call_unary_numeric_func<float>([](float x)->float { return std::cos(x); }, arg, filter);
            return res;
        }

        static ZfxVariable sinh(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const auto& arg = args[0];
            ZfxVariable res = call_unary_numeric_func<float>([](float x)->float { return std::sinh(x); }, arg, filter);
            return res;
        }

        static ZfxVariable cosh(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const auto& arg = args[0];
            ZfxVariable res = call_unary_numeric_func<float>([](float x)->float { return std::cosh(x); }, arg, filter);
            return res;
        }

        static ZfxVariable fit01(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 3) throw makeError<UnimplError>();

            //只考虑单值的情况: fit01(.3,5,20)=9.5
            float value = get_zfxvec_front_elem<float>(args[0].value);
            float omin = 0;
            float omax = 1;
            float nmin = get_zfxvec_front_elem<float>(args[1].value);
            float nmax = get_zfxvec_front_elem<float>(args[2].value);

            if (nmin == nmax) {
                throw makeError<UnimplError>("the omin == omax or nmin == nmax");
            }
            if (value < omin || value >omax) {
                throw makeError<UnimplError>("the value is not between 0 and 1, when calling `fit01`");
            }
            float mp_value = ((value - omin) / (omax - omin)) * (nmax - nmin) + nmin;
            ZfxVariable ret;
            ret.value = std::vector<float>{ mp_value };
            return ret;
        }

        static ZfxVariable fit(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 5) throw makeError<UnimplError>();

            //只考虑单值的情况: fit(.3, 0, 1, 10, 20) == 13
            float value = get_zfxvec_front_elem<float>(args[0].value);
            float omin = get_zfxvec_front_elem<float>(args[1].value);
            float omax = get_zfxvec_front_elem<float>(args[2].value);
            float nmin = get_zfxvec_front_elem<float>(args[3].value);
            float nmax = get_zfxvec_front_elem<float>(args[4].value);

            if (omin == omax || nmin == nmax) {
                throw makeError<UnimplError>("the omin == omax or nmin == nmax");
            }
            if (value < omin || value >omax) {
                throw makeError<UnimplError>("the value is not between omin and omax, when calling `fit`");
            }
            float mp_value = ((value - omin) / (omax - omin)) * (nmax - nmin) + nmin;
            ZfxVariable ret;
            ret.value = std::vector<float>(mp_value);
            return ret;
        }

        static ZfxVariable rand(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() > 1) throw makeError<UnimplError>();
            int N = 1;
            ZfxVariable res;
            if (args.size() == 1) {
                res = call_unary_numeric_func<float>([](float arg)->float {
                    std::mt19937 rng(arg);
                    std::uniform_real_distribution<float> dist(0.0, 1.0);
                    return dist(rng);
                    }, args[0], filter);
            }
            else {
                //没有参数作为随机数种子
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> dist(0.0, 1.0);
                res.value = std::vector<float>{ (dist(rng)) };
            }
            return res;
        }

        static ZfxVariable pow(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>();
            const auto& arg = args[0];
            const auto& idx = args[1];

            return std::visit([&](const auto& arg_vec, const auto& idx_vec)->ZfxVariable {
                using T1 = std::decay_t<decltype(arg_vec)>;
                using E1 = typename T1::value_type;
                using T2 = std::decay_t<decltype(idx_vec)>;
                using E2 = typename T2::value_type;

                if constexpr ((std::is_same_v<E1, float> || std::is_same_v<E1, int>) &&
                    (std::is_same_v<E2, float> || std::is_same_v<E2, int>))
                {
                    ZfxVariable res;
                    std::vector<float> resvec(arg_vec.size());
                    for (int i = 0; i < arg_vec.size(); i++) {
                        resvec[i] = std::pow(arg_vec[i], idx_vec[0]);
                    }
                    res.value = std::move(resvec);
                    return res;
                }
                else {
                    throw makeError<UnimplError>("not support type in `pow`");
                }
                }, args[0].value, args[1].value);
        }

        static ZfxVariable add_point(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() == 1) {
                const auto& arg = args[0];
                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                    int ptnum = -1;
                    std::visit([&](auto&& vec) {
                        using T = std::decay_t<decltype(vec)>;
                        using E = typename T::value_type;
                        //暂时只考虑一个点
                        const auto& val = vec[0];

                        if constexpr (std::is_same_v<E, glm::vec3> || std::is_same_v<E, zeno::vec3f> || std::is_same_v<E, zeno::vec3i>) {
                            ptnum = spGeo->m_impl->add_point(zeno::vec3f(val[0], val[1], val[2]));
                        } else if constexpr (std::is_same_v<E, zfxintarr> || std::is_same_v<E, zfxfloatarr>) {
                            if (val.size() == 3) {
                                ptnum = spGeo->m_impl->add_point(zeno::vec3f(val[0], val[1], val[2]));
                            } else {
                                throw makeError<UnimplError>("the number of arguments of add_point is not matched.");
                            }
                        } else {
                            throw makeError<UnimplError>("the type of arguments of add_point is not matched.");
                        }
                        }, arg.value);
                    ZfxVariable res;
                    res.value = std::vector<int>{ ptnum };
                    return res;
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
            else if (args.empty()) {
                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                    //暂时只考虑一个点
                    int ptnum = spGeo->m_impl->add_point(zeno::vec3f(0, 0, 0));
                    ZfxVariable res;
                    res.value = std::vector<int>{ ptnum };
                    return res;
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
            else {
                throw makeError<UnimplError>();
            }
        }

        static ZfxVariable add_vertex(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2) {
                throw makeError<UnimplError>();
            }
            if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                int faceid = get_zfxvec_front_elem<int>(args[0].value);
                int pointid = get_zfxvec_front_elem<int>(args[1].value);
                int vertid = spGeo->m_impl->add_vertex(faceid, pointid);
                ZfxVariable res;
                res.value = std::vector<int>{ vertid };
                return res;
            }
            else {
                throw makeError<UnimplError>();
            }
        }

        static ZfxVariable remove_vertex(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of remove_vertex is not matched.");

            const auto& arg = args[0];
            int N = arg.size();
            if (N == 0)
                return initVarFromZvar(false);

            const auto& arg2 = args[1];
            N = arg2.size();
            if (N == 0)
                return initVarFromZvar(false);

            bool bSucceed = false;

            if (N < filter.size()) {
                assert(N == 1);
                int faceid = get_zfxvec_front_elem<int>(arg.value);
                int vertid = get_zfxvec_front_elem<int>(arg2.value);

                /* 删除pointnum的点，如果成功，就返回原来下一个点的pointnum(应该就是pointnum)，失败就返回-1 */
                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                    bSucceed = spGeo->m_impl->remove_vertex(faceid, vertid);
                }

                if (bSucceed) {
                } else {
                    throw makeError<UnimplError>("error on removeVertex");
                }
#if 0
                if (bSucceed) {
                    //要调整filter，移除掉第currrem位置的元素
                    filter.erase(filter.begin() + currrem);
                    //所有储存在m_globalAttrCached里的属性都移除第currrem号元素，如果有ptnum，也要调整

                    //移除点以后要调整已有的属性值
                    for (auto& [name, attrVar] : *pContext->zfxVariableTbl) {
                        auto& attrvalues = attrVar.value;
                        if (name == "@ptnum") {
                            assert(currrem < attrvalues.size());
                            for (int i = currrem + 1; i < attrvalues.size(); i++)
                                attrvalues[i] = i - 1;
                            attrvalues.erase(attrvalues.begin() + currrem);
                        }
                        else {
                            attrvalues.erase(attrvalues.begin() + currrem);
                        }
                    }
                }
                else {
                    throw makeError<UnimplError>("error on removePoint");
                }
#endif
            }
            else {
                //移除多个的情况
                //TODO:
            }
            return initVarFromZvar(bSucceed);
        }

        static bool _remove_points(std::deque<int> remPoints, ZfxElemFilter& filter, ZfxContext* pContext) {
            bool bSucceed = false;
            while (!remPoints.empty())
            {
                int currrem = remPoints.front();
                remPoints.pop_front();
                bSucceed = false;
                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                    bSucceed = spGeo->m_impl->remove_point(currrem);
                }
                if (bSucceed) {
                    //要调整filter，移除掉第currrem位置的元素
                    filter.erase(filter.begin() + currrem);
                    //所有储存在m_globalAttrCached里的属性都移除第currrem号元素，如果有ptnum，也要调整
                    //afterRemovePoint(currrem);
                    for (auto& [name, attrVar] : *pContext->zfxVariableTbl) {
                        auto& attrvalues = attrVar.value;
                        bool isptnum = name == "@ptnum";

                        std::visit([&](auto& vec) {
                            using T = std::decay_t<decltype(vec)>;
                            using E = typename T::value_type;

                            if (isptnum) {
                                assert(currrem < vec.size());
                                if constexpr (std::is_same_v<E, int>) {
                                    for (int i = currrem + 1; i < vec.size(); i++)
                                        vec[i] = i - 1;
                                    vec.erase(vec.begin() + currrem);
                                }
                            }
                            else {
                                vec.erase(vec.begin() + currrem);
                            }
                            }, attrvalues);
                    }

                    //最后将当前所有剩下的删除点的序号再自减
                    for (auto iter = remPoints.begin(); iter != remPoints.end(); iter++) {
                        *iter -= 1;
                    }
                }
                else {
                    throw makeError<UnimplError>("error on removePoint");
                }
            }
            return bSucceed;
        }

        static ZfxVariable remove_point(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of remove_point is not matched.");

            ZfxVariable arg = args[0];
            int N = arg.size();
            if (N == 0) return initVarFromZvar(false);
            bool bSucceed = false;

            if (N == 1) {
                //可能是数组
                std::visit([&](auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    if constexpr (std::is_same_v<E, zfxintarr>) {
                        const auto& arr = vec[0];
                        std::deque<int> remPoints;
                        for (int pt : arr) {
                            remPoints.push_back(pt);
                        }
                        bSucceed = _remove_points(remPoints, filter, pContext);
                    }
                    else {
                        bSucceed = false;
                    }
                    }, arg.value);

                if (bSucceed)
                    return initVarFromZvar(bSucceed);
            }

            if (N < filter.size()) {
                assert(N == 1);
                int currrem = get_zfxvec_front_elem<int>(arg.value);

                /* 删除pointnum的点，如果成功，就返回原来下一个点的pointnum(应该就是pointnum)，失败就返回-1 */
                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
                    bSucceed = spGeo->m_impl->remove_point(currrem);
                }
                if (bSucceed) {
                    //要调整filter，移除掉第currrem位置的元素
                    filter.erase(filter.begin() + currrem);
                    //所有储存在m_globalAttrCached里的属性都移除第currrem号元素，如果有ptnum，也要调整

                    //移除点以后要调整已有的属性值
                    for (auto& [name, attrVar] : *pContext->zfxVariableTbl) {
                        auto& attrvalues = attrVar.value;
                        bool isptnum = name == "@ptnum";

                        std::visit([&](auto& vec) {
                            using T = std::decay_t<decltype(vec)>;
                            using E = typename T::value_type;

                            if (isptnum) {
                                assert(currrem < vec.size());
                                if constexpr (std::is_same_v<E, int>)
                                {
                                    for (int i = currrem + 1; i < vec.size(); i++)
                                        vec[i] = i - 1;
                                    vec.erase(vec.begin() + currrem);
                                }
                            }
                            else {
                                vec.erase(vec.begin() + currrem);
                            }
                            }, attrvalues);
                    }
                }
                else {
                    throw makeError<UnimplError>("error on removePoint");
                }
            }
            else {
                std::deque<int> remPoints;
                assert(N == filter.size());

                std::visit([&](auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                        for (int i = 0; i < N; i++) {
                            if (!filter[i]) continue;
                            int pointnum = vec[i];
                            remPoints.push_back(pointnum);
                        }
                    }
                    else {
                        throw makeError<UnimplError>("not support type");
                    }
                    }, arg.value);

                bSucceed = _remove_points(remPoints, filter, pContext);
            }
            return initVarFromZvar(bSucceed);
        }

        static ZfxVariable add_face(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of add_face is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            const auto& points = get_zfxvec_front_elem<std::vector<int>>(args[0].value);
            int ret = spGeo->m_impl->add_face(points);
            return initVarFromZvar(ret);
        }

        static ZfxVariable remove_face(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() > 2 || args.empty())
                throw makeError<UnimplError>("the number of arguments of remove_face is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            const auto& arg = args[0];
            bool bIncludePoints = true;
            if (args.size() == 2) {
                bIncludePoints = get_zfxvec_front_elem<bool>(args[1].value);
            }

            int N = arg.size();
            if (N == 0) return initVarFromZvar(false);
            bool bSucceed = false;

            std::set<int> remfaces;
            if (N < filter.size()) {
                assert(N == 1);

                std::visit([&](const auto& vec) {//N为1的时候也可能是一个std::vector<int>，需判断
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;

                    if constexpr (std::is_same_v<E, int>) {
                        int currrem = vec[0];
                        remfaces.insert(currrem);
                    }
                    else if constexpr (std::is_same_v<E, float>) {
                        int currrem = static_cast<int>(vec[0]);
                        remfaces.insert(currrem);
                    }
                    else if constexpr (std::is_same_v<E, zfxintarr>) {
                        for (const auto& i : vec[0]) {
                            remfaces.insert(i);
                        }
                    }
                }, arg.value);
            }
            else {
                assert(N == filter.size());
                std::visit([&](const auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;

                    if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                        for (int i = 0; i < vec.size(); i++) {
                            if (!filter[i]) continue;
                            int pointnum = vec[i];
                            remfaces.insert(pointnum);
                        }
                    }
                    }, arg.value);
            }

            bSucceed = spGeo->m_impl->remove_faces(remfaces, bIncludePoints);
            if (bSucceed) {
                //要调整filter，移除掉第currrem位置的元素
                removeElemsByIndice(filter, remfaces);
                //afterRemoveElements(remfaces);
                for (auto& [name, attrVar] : *pContext->zfxVariableTbl) {
                    auto& attrvalues = attrVar.value;

                    std::visit([&](auto& vec) {
                        removeElemsByIndice(vec, remfaces);
                    }, attrVar.value);
                }
            }
            else {
                throw makeError<UnimplError>("error on removeface");
            }
            return initVarFromZvar(bSucceed);
        }

        static ZfxVariable create_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 3)
                throw makeError<UnimplError>("the number of arguments of create_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());

            std::string group = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string name = get_zfxvec_front_elem<std::string>(args[1].value);

            GeoAttrGroup grp = ATTR_POINT;
            if (group == "vertex") grp = ATTR_VERTEX;
            else if (group == "point") grp = ATTR_POINT;
            else if (group == "face") grp = ATTR_FACE;
            else if (group == "geometry") grp = ATTR_GEO;

            AttrVar defl = zfxVarToAttrVar(args[2]);
            int ret = spGeo->m_impl->create_attr(grp, name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable create_face_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of create_face_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->create_face_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable create_point_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of create_point_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->create_point_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable create_vertex_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of create_vertex_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->create_vertex_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable create_geometry_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of create_geometry_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->create_geometry_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable set_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 3)
                throw makeError<UnimplError>("the number of arguments of set_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());

            std::string group = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string name = get_zfxvec_front_elem<std::string>(args[1].value);

            GeoAttrGroup grp = ATTR_POINT;
            if (group == "vertex") grp = ATTR_VERTEX;
            else if (group == "point") grp = ATTR_POINT;
            else if (group == "face") grp = ATTR_FACE;
            else if (group == "geometry") grp = ATTR_GEO;

            AttrVar defl = zfxVarToAttrVar(args[2]);
            int ret = spGeo->m_impl->set_attr(grp, name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable set_vertex_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of set_vertex_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->set_vertex_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable set_point_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of set_point_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->set_point_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable set_face_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of set_face_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->set_face_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable set_geometry_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of set_geometry_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            AttrVar defl = zfxVarToAttrVar(args[1]);
            int ret = spGeo->m_impl->set_geometry_attr(name, defl);
            return initVarFromZvar(ret);
        }

        static ZfxVariable has_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of has_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string group = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string name = get_zfxvec_front_elem<std::string>(args[1].value);

            GeoAttrGroup grp = ATTR_POINT;
            if (group == "vertex") grp = ATTR_VERTEX;
            else if (group == "point") grp = ATTR_POINT;
            else if (group == "face") grp = ATTR_FACE;
            else if (group == "geometry") grp = ATTR_GEO;

            bool hasattr = spGeo->m_impl->has_attr(grp, name);
            return initVarFromZvar(hasattr);
        }

        static ZfxVariable has_vertex_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of has_vertex_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            bool hasattr = spGeo->m_impl->has_vertex_attr(name);
            return initVarFromZvar(hasattr);
        }

        static ZfxVariable has_point_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of has_point_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            bool hasattr = spGeo->m_impl->has_point_attr(name);
            return initVarFromZvar(hasattr);
        }

        static ZfxVariable has_face_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of has_vertex_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            bool hasattr = spGeo->m_impl->has_face_attr(name);
            return initVarFromZvar(hasattr);
        }

        static ZfxVariable has_geometry_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of has_geometry_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            bool hasattr = spGeo->m_impl->has_geometry_attr(name);
            return initVarFromZvar(hasattr);
        }

        static ZfxVariable delete_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of delete_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string group = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string name = get_zfxvec_front_elem<std::string>(args[1].value);

            GeoAttrGroup grp = ATTR_POINT;
            if (group == "vertex") grp = ATTR_VERTEX;
            else if (group == "point") grp = ATTR_POINT;
            else if (group == "face") grp = ATTR_FACE;
            else if (group == "geometry") grp = ATTR_GEO;

            int ret = spGeo->m_impl->delete_attr(grp, name);
            return initVarFromZvar(ret);
        }

        static ZfxVariable delete_vertex_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of delete_vertex_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            int ret = spGeo->m_impl->delete_vertex_attr(name);
            return initVarFromZvar(ret);
        }

        static ZfxVariable delete_point_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of delete_point_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            int ret = spGeo->m_impl->delete_point_attr(name);
            return initVarFromZvar(ret);
        }

        static ZfxVariable delete_face_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of delete_face_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            int ret = spGeo->m_impl->delete_face_attr(name);
            return initVarFromZvar(ret);
        }

        static ZfxVariable delete_geometry_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of delete_geometry_attr is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            int ret = spGeo->m_impl->delete_geometry_attr(name);
            return initVarFromZvar(ret);
        }

        static ZfxVariable npoints(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() > 1) {
                throw makeError<UnimplError>("the number of arguments of npoints is not matched.");
            } else if (args.size() == 1) {
                std::string ref = get_zfxvec_front_elem<std::string>(args[0].value);
                if (std::regex_search(ref, FunctionManager::refPattern)) {
                    auto spObj = getObjFromRef(ref, pContext);
                    if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(spObj)) {
                        return initVarFromZvar(spGeo->m_impl->npoints());
                    } else {
                        throw makeError<UnimplError>("npoints function refers an empty output object.");
                    }
                } else {
                    throw makeError<UnimplError>("npoints can not resolve ref.");
                }
            } else {
                auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                int ret = spGeo->m_impl->npoints();
                return initVarFromZvar(ret);
            }
        }

        static ZfxVariable nfaces(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 0)
                throw makeError<UnimplError>("the number of arguments of nfaces is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int ret = spGeo->m_impl->nfaces();
            return initVarFromZvar(ret);
        }

        static ZfxVariable nvertices(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 0)
                throw makeError<UnimplError>("the number of arguments of nvertices is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int ret = spGeo->m_impl->nvertices();
            return initVarFromZvar(ret);
        }

        /* 点相关 */
        static ZfxVariable point_faces(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of point_faces is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int pointid = get_zfxvec_front_elem<int>(args[0].value);
            std::vector<int> ret = spGeo->m_impl->point_faces(pointid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable point_vertex(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of point_vertex is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int pointid = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->point_vertex(pointid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable point_vertices(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of point_vertices is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int pointid = get_zfxvec_front_elem<int>(args[0].value);
            std::vector<int> ret = spGeo->m_impl->point_vertices(pointid);
            return initVarFromZvar(ret);
        }

        /* 面相关 */
        static ZfxVariable face_point(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of face_point is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int faceid = get_zfxvec_front_elem<int>(args[0].value);
            int vertid = get_zfxvec_front_elem<int>(args[1].value);
            int ret = spGeo->m_impl->face_point(faceid, vertid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable face_points(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of face_points is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            return call_unary_numeric_func<std::vector<int>>([&](int faceid)->std::vector<int> {
                return spGeo->m_impl->face_points(faceid);
                }, args[0], filter);
        }

        static ZfxVariable face_vertex(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of face_vertex is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int faceid = get_zfxvec_front_elem<int>(args[0].value);
            int vertid = get_zfxvec_front_elem<int>(args[1].value);
            int ret = spGeo->m_impl->face_vertex(faceid, vertid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable face_vertex_count(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of face_vertex_count is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int faceid = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->face_vertex_count(faceid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable face_vertices(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of face_vertices is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int face_id = get_zfxvec_front_elem<int>(args[0].value);
            std::vector<int> ret = spGeo->m_impl->face_vertices(face_id);
            return initVarFromZvar(ret);
        }

        /* Vertex相关 */
        static ZfxVariable vertex_index(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of vertex_index is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int faceid = get_zfxvec_front_elem<int>(args[0].value);
            int vertid = get_zfxvec_front_elem<int>(args[1].value);
            int ret = spGeo->m_impl->vertex_index(faceid, vertid);
            return initVarFromZvar(ret);
        }

        static ZfxVariable vertex_next(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of vertex_next is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int linear_vertex_id = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->vertex_next(linear_vertex_id);
            return initVarFromZvar(ret);
        }

        static ZfxVariable vertex_prev(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of vertex_prev is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int linear_vertex_id = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->vertex_prev(linear_vertex_id);
            return initVarFromZvar(ret);
        }

        static ZfxVariable vertex_point(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of vertex_point is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int linear_vertex_id = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->vertex_point(linear_vertex_id);
            return initVarFromZvar(ret);
        }

        static ZfxVariable vertex_face(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of vertex_face is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int linear_vertex_id = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->vertex_face(linear_vertex_id);
            return initVarFromZvar(ret);
        }

        static ZfxVariable vertex_face_index(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of vertex_face_index is not matched.");

            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            int linear_vertex_id = get_zfxvec_front_elem<int>(args[0].value);
            int ret = spGeo->m_impl->vertex_face_index(linear_vertex_id);
            return initVarFromZvar(ret);
        }

        static ZfxVariable get_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of get_attr is not matched.");

            std::string group = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string name = get_zfxvec_front_elem<std::string>(args[1].value);

            GeoAttrGroup grp = ATTR_POINT;
            if (group == "vertex") grp = ATTR_VERTEX;
            else if (group == "point") grp = ATTR_POINT;
            else if (group == "face") grp = ATTR_FACE;
            else if (group == "geometry") grp = ATTR_GEO;
            
            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            assert(spGeo);
            ZfxVariable var;
            var.value = getAttrs(spGeo->m_impl.get(), grp, name);
            return var;
        }

        static ZfxVariable get_vertex_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of get_vertex_attr is not matched.");

            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            assert(spGeo);
            ZfxVariable var;
            var.value = getAttrs(spGeo->m_impl.get(), ATTR_VERTEX, name);
            return var;
        }

        static ZfxVariable get_point_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() > 2)
                throw makeError<UnimplError>("the number of arguments of get_point_attr is not matched.");

            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            assert(spGeo);

            ZfxVariable ret;
            const auto& attrs = getAttrs(spGeo->m_impl.get(), ATTR_POINT, name);
            if (args.size() == 2) {
                //取索引
                //TODO: 调getElem，就不用整个attrs都拿出来，因为可能在循环下调用的
                std::visit([&](const auto& vec, const auto& idx_vec) {
                    using T1 = std::decay_t<decltype(vec)>;
                    using E1 = typename T1::value_type;
                    using T2 = std::decay_t<decltype(idx_vec)>;
                    using E2 = typename T2::value_type;

                    if (idx_vec.size() > vec.size()) {
                        throw makeError<UnimplError>("the attr size doesn't match");
                    }

                    if constexpr (std::is_same_v<E2, int> || std::is_same_v<E2, float>) {
                        const int N = idx_vec.size();
                        std::vector<E1> ret_vec(N);
                        for (int i = 0; i < N; i++) {
                            int idx = idx_vec[i];
                            ret_vec[i] = vec[i];
                        }
                        ret.value = std::move(ret_vec);
                    }
                    else {
                        throw makeError<UnimplError>("not support type");
                    }
                    }, attrs, args[1].value);
            }
            else {
                ret.value = attrs;
            }
            return ret;
        }

        static ZfxVariable get_face_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of get_face_attr is not matched.");

            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            ZfxVariable var;
            if (args.size() == 2) {
                //TODO:
                throw makeError<UnimplError>("unimpl case in `get_face_attr`");
            }
            else {
                var.value = getAttrs(spGeo->m_impl.get(), ATTR_FACE, name);
            }
            return var;
        }

        static ZfxVariable get_geometry_attr(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 1)
                throw makeError<UnimplError>("the number of arguments of get_geometry_attr is not matched.");

            std::string name = get_zfxvec_front_elem<std::string>(args[0].value);
            auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
            ZfxVariable var;
            var.value = getAttrs(spGeo->m_impl.get(), ATTR_GEO, name);
            return var;
        }

        static ZfxVariable bbox(const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
            if (args.size() != 2)
                throw makeError<UnimplError>("the number of arguments of get_geometry_attr is not matched.");

            std::string nodepath = get_zfxvec_front_elem<std::string>(args[0].value);
            std::string type = get_zfxvec_front_elem<std::string>(args[1].value);

            std::string parampath, _;
            auto spNode = zfx::getNodeAndParamFromRefString(nodepath, pContext, _, parampath);
            CalcContext ctx;
            spNode->execute(&ctx);
            auto obj = getObjFromRef(nodepath, pContext);
            int res = 0;
            return initVarFromZvar(res);
        }

        ZfxVariable callFunction(const std::string& funcname, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext)
        {
            if (funcname == "ref") {
                return zeno::zfx::callRef(args, filter, pContext);
            }
            if (funcname == "param" || funcname == "parameter") {
                return parameter(args, filter, pContext);
            }
            if (funcname == "log") {
                return zeno::zfx::log(args, filter, pContext);
            }
            if (funcname == "vec3") {
                return zeno::zfx::vec3(args, filter, pContext);
            }
            if (funcname == "sin") {
                return zeno::zfx::sin(args, filter, pContext);
            }
            if (funcname == "cos") {
                return zeno::zfx::cos(args, filter, pContext);
            }
            if (funcname == "sinh") {
                return sinh(args, filter, pContext);
            }
            if (funcname == "cosh") {
                return cosh(args, filter, pContext);
            }
            if (funcname == "rand") {
                return rand(args, filter, pContext);
            }
            if (funcname == "pow") {
                return pow(args, filter, pContext);
            }
            if (funcname == "create_attr") {
                return create_attr(args, filter, pContext);
            }
            if (funcname == "create_face_attr") {
                return create_face_attr(args, filter, pContext);
            }
            if (funcname == "create_point_attr") {
                return create_point_attr(args, filter, pContext);
            }
            if (funcname == "create_vertex_attr") {
                return create_vertex_attr(args, filter, pContext);
            }
            if (funcname == "create_geometry_attr") {
                return create_geometry_attr(args, filter, pContext);
            }
            if (funcname == "set_attr") {
                return set_attr(args, filter, pContext);
            }
            if (funcname == "set_vertex_attr") {
                return set_vertex_attr(args, filter, pContext);
            }
            if (funcname == "set_point_attr") {
                return set_point_attr(args, filter, pContext);
            }
            if (funcname == "set_face_attr") {
                return set_face_attr(args, filter, pContext);
            }
            if (funcname == "set_geometry_attr") {
                return set_geometry_attr(args, filter, pContext);
            }
            if (funcname == "has_attr") {
                return has_attr(args, filter, pContext);
            }
            if (funcname == "has_vertex_attr") {
                return has_vertex_attr(args, filter, pContext);
            }
            if (funcname == "has_point_attr") {
                return has_point_attr(args, filter, pContext);
            }
            if (funcname == "has_face_attr") {
                return create_attr(args, filter, pContext);
            }
            if (funcname == "has_geometry_attr") {
                return has_geometry_attr(args, filter, pContext);
            }
            if (funcname == "delete_attr") {
                return delete_attr(args, filter, pContext);
            }
            if (funcname == "delete_vertex_attr") {
                return delete_vertex_attr(args, filter, pContext);
            }
            if (funcname == "delete_point_attr") {
                return delete_point_attr(args, filter, pContext);
            }
            if (funcname == "delete_face_attr") {
                return delete_face_attr(args, filter, pContext);
            }
            if (funcname == "delete_geometry_attr") {
                return delete_geometry_attr(args, filter, pContext);
            }
            if (funcname == "add_vertex") {
                return add_vertex(args, filter, pContext);
            }
            if (funcname == "add_point") {
                return add_point(args, filter, pContext);
            }
            if (funcname == "add_face") {
                return add_face(args, filter, pContext);
            }
            if (funcname == "remove_face") {
                return remove_face(args, filter, pContext);
            }
            if (funcname == "remove_point") {
                return remove_point(args, filter, pContext);
            }
            if (funcname == "remove_vertex") {
                return remove_vertex(args, filter, pContext);
            }
            if (funcname == "npoints") {
                return npoints(args, filter, pContext);
            }
            if (funcname == "nfaces") {
                return nfaces(args, filter, pContext);
            }
            if (funcname == "nvertices") {
                return nvertices(args, filter, pContext);
            }
            if (funcname == "point_faces") {
                return point_faces(args, filter, pContext);
            }
            if (funcname == "point_vertex") {
                return point_vertex(args, filter, pContext);
            }
            if (funcname == "point_vertices") {
                return point_vertices(args, filter, pContext);
            }
            if (funcname == "face_point") {
                return face_point(args, filter, pContext);
            }
            if (funcname == "face_points") {
                return face_points(args, filter, pContext);
            }
            if (funcname == "face_vertex") {
                return face_vertex(args, filter, pContext);
            }
            if (funcname == "face_vertex_count") {
                return face_vertex_count(args, filter, pContext);
            }
            if (funcname == "face_vertices") {
                return face_vertices(args, filter, pContext);
            }
            if (funcname == "vertex_index") {
                return vertex_index(args, filter, pContext);
            }
            if (funcname == "vertex_next") {
                return vertex_next(args, filter, pContext);
            }
            if (funcname == "vertex_prev") {
                return vertex_prev(args, filter, pContext);
            }
            if (funcname == "vertex_point") {
                return vertex_point(args, filter, pContext);
            }
            if (funcname == "vertex_face") {
                return vertex_face(args, filter, pContext);
            }
            if (funcname == "vertex_face_index") {
                return vertex_face_index(args, filter, pContext);
            }
            if (funcname == "get_attr") {
                return get_attr(args, filter, pContext);
            }            
            if (funcname == "get_vertex_attr") {
                return get_vertex_attr(args, filter, pContext);
            }            
            if (funcname == "get_point_attr") {
                return get_point_attr(args, filter, pContext);
            }
            if (funcname == "get_face_attr") {
                return get_face_attr(args, filter, pContext);
            }
            if (funcname == "get_geometry_attr") {
                return get_geometry_attr(args, filter, pContext);
            }
            if (funcname == "bbox") {
                return bbox(args, filter, pContext);
            }
            if (funcname == "rint" || funcname == "round") {
                return call_unary_numeric_func<float>([&](float val)->float {
                    return std::round(val);
                    }, args[0], filter);
            }
            if (funcname == "fit") {
                return fit(args, filter, pContext);
            }
            if (funcname == "fit01") {
                return fit01(args, filter, pContext);
            }
            if (funcname == "append") {
                const ZfxVariable& arr_var = args[0];
                const ZfxVariable& elem_var = args[1];
                const int N = arr_var.size();
                if (N != elem_var.size()) {
                    throw makeError<UnimplError>("the size of array and element doesn't match, when calling `append`");
                }

                ZfxVariable ret;    //引用机制实现有点麻烦，又得回填到局部变量表，目前先拷贝返回
                ret = arr_var;

                std::visit([&](auto& arr_vec, auto& elem_vec) {
                    using T1 = std::decay_t<decltype(arr_vec)>;
                    using E1 = typename T1::value_type;
                    using T2 = std::decay_t<decltype(elem_vec)>;
                    using E2 = typename T2::value_type;

                    constexpr bool is_arr = std::is_same_v<E1, zfxintarr> ||
                        std::is_same_v<E1, zfxfloatarr> ||
                        std::is_same_v<E1, zfxstringarr> ||
                        std::is_same_v<E1, zfxvec2arr> ||
                        std::is_same_v<E1, zfxvec3arr> ||
                        std::is_same_v<E1, zfxvec4arr>;

                    if constexpr (is_arr) {
                        using ElemType = typename E1::value_type;
                        if constexpr (std::is_same_v<ElemType, E2>) {
                            for (int i = 0; i < arr_vec.size(); i++) {
                                arr_vec[i].push_back((ElemType)elem_vec[i]);
                            }
                        }
                    }
                    else {
                        throw makeError<UnimplError>("only support array to append");
                    }
                    }, ret.value, elem_var.value);

                return ret;
            }
            if (funcname == "len") {
                return std::visit([&](auto& vec)->ZfxVariable {
                    using T1 = std::decay_t<decltype(vec)>;
                    using E1 = typename T1::value_type;

                    constexpr bool is_arr = std::is_same_v<E1, zfxintarr> ||
                        std::is_same_v<E1, zfxfloatarr> ||
                        std::is_same_v<E1, zfxstringarr> ||
                        std::is_same_v<E1, zfxvec2arr> ||
                        std::is_same_v<E1, zfxvec3arr> ||
                        std::is_same_v<E1, zfxvec4arr>;

                    if constexpr (is_arr) {
                        ZfxVariable ret;
                        std::vector<int> szvec(vec.size());
                        for (int i = 0; i < vec.size(); i++) {
                            szvec[i] = vec[i].size();
                        }
                        ret.value = std::move(szvec);
                        return ret;
                    }
                    else {
                        throw makeError<UnimplError>("only support array to append");
                    }
                    }, args[0].value);
            }
            if (funcname == "get_bboxmin") {
                auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                assert(spGeo);
                std::pair<vec3f, vec3f> ret = geomBoundingBox(spGeo->m_impl.get());
                glm::vec3 bmin(ret.first[0], ret.first[1], ret.first[2]);
                return initVarFromZvar(bmin);
            }
            if (funcname == "bboxmin") {
                const std::string& nodename = get_zfxvec_front_elem<std::string>(args[0].value);
                NodeImpl* pObjNode = pContext->spNode->getGraph()->getNode(nodename);
                if (!pObjNode) {
                    throw makeError<UnimplError>("unknown node `" + nodename + "`");
                }
                auto targetObj = pObjNode->get_default_output_object();
                if (!targetObj) {
                    throw makeError<UnimplError>("get nullptr obj from default output when `prim_has_attr` is called");
                }

                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(targetObj)) {
                    std::pair<vec3f, vec3f> ret = geomBoundingBox(spGeo->m_impl.get());
                    glm::vec3 bmin(ret.first[0], ret.first[1], ret.first[2]);
                    return initVarFromZvar(bmin);
                }
                else {
                    throw makeError<UnimplError>("get not geometry obj when calling `prim_has_attr`");
                }
            }
            if (funcname == "get_bboxmax") {
                auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                assert(spGeo);
                std::pair<vec3f, vec3f> ret = geomBoundingBox(spGeo->m_impl.get());
                glm::vec3 bmax(ret.second[0], ret.second[1], ret.second[2]);
                return initVarFromZvar(bmax);
            }
            if (funcname == "pcopen") {
                std::string inputgeo = get_zfxvec_front_elem<std::string>(args[0].value);
                std::string attrname = get_zfxvec_front_elem<std::string>(args[1].value);
                float radius = get_zfxvec_front_elem<float>(args[3].value);
                int maxpoints = get_zfxvec_front_elem<int>(args[4].value);
                if (attrname != "P") {
                    throw makeError<UnimplError>("only support Pos as the base attribute of point clound");
                }

                std::vector<vec3f> points;
                if (inputgeo == "0") {
                    //this geometry.
                    auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                    assert(spGeo);
                    points = spGeo->m_impl->points_pos();
                }
                else {
                    //todo: other geometry
                }

                PointCloud pc;
                pc.radius = radius;
                pc.maxpoints = maxpoints;
                pc.pTree = std::make_shared<zeno::KdTree>(points, points.size());

                std::visit([&](const auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    if constexpr (std::is_same_v<E, glm::vec3>) {
                        pc.testPoints.resize(vec.size());
                        for (int i = 0; i < pc.testPoints.size(); i++) {
                            const auto& _pt = vec[i];
                            pc.testPoints[i] = zeno::vec3f(_pt[0], _pt[1], _pt[2]);
                        }
                    }
                    else {
                        throw makeError<UnimplError>("not support type");
                    }
                    }, args[3].value);

                int handle = pContext->pchandles.size();
                pContext->pchandles.push_back(pc);
                return initVarFromZvar(handle);
            }
            if (funcname == "pcnumfound") {
                int handle = get_zfxvec_front_elem<int>(args[0].value);
                if (handle < 0 || handle >= pContext->pchandles.size()) {
                    throw makeError<UnimplError>("invalid pchandle");
                }
                else {
                    PointCloud& pc = pContext->pchandles[handle];
                    std::vector<int> pcnumfind;
                    for (int i = 0; i < pc.testPoints.size(); i++) {
                        zeno::vec3f pt = pc.testPoints[i];
                        std::set<int> pts = pc.pTree->fix_radius_search(pt, pc.radius);
                        pcnumfind.push_back(pts.size());
                    }
                    ZfxVariable ret;
                    ret.value = std::move(pcnumfind);
                    return ret;
                }
            }
            if (funcname == "primreduce") {
                auto _attrName = get_zfxvec_front_elem<std::string>(args[0].value);
                zeno::String attrName = stdString2zs(_attrName);
                auto op = get_zfxvec_front_elem<std::string>(args[1].value);
                auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                zeno::GeoAttrType type = spGeo->get_attr_type(ATTR_POINT, attrName);
                zeno::NumericValue result;
                if (zeno::ATTR_FLOAT == type) {
                    const std::vector<float>& attrData = spGeo->get_float_attr(ATTR_POINT, attrName);
                    result = prim_reduce(attrData, op);
                }
                else if (zeno::ATTR_VEC3 == type) {
                    const std::vector<zeno::vec3f>& attrData = spGeo->get_vec3f_attr(ATTR_POINT, attrName);
                    result = prim_reduce(attrData, op);
                }
                else {
                    throw makeError<UnimplError>("attr type unknown when calling primreduce");
                }
                ZfxVariable ret;
                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, int>) {
                        ret = initVarFromZvar(arg);
                    }
                    else if constexpr (std::is_same_v<T, float>) {
                        ret = initVarFromZvar(arg);
                    }
                    else if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        ret = initVarFromZvar(glm::vec3(arg[0], arg[1], arg[2]));
                    }
                    else {
                        throw makeError<UnimplError>("unknown type");
                    }
                }, result);
                return ret;
            }
            if (funcname == "abs") {
                const ZfxVariable& var = args[0];
                return call_unary_numeric_func<float>([](float val)->float {return std::abs(val); }, args[0], filter);
            }
            if (funcname == "sqrt") {
                const ZfxVariable& var = args[0];
                return call_unary_numeric_func<float>([](float val)->float {return std::sqrt(val); }, args[0], filter);
            }
            if (funcname == "cross") {
                auto res = std::visit([&](auto&& v1, auto&& v2)->glm::vec3 {
                    using T1 = std::decay_t<decltype(v1)>;
                    using E1 = typename T1::value_type;
                    using T2 = std::decay_t<decltype(v2)>;
                    using E2 = typename T2::value_type;

                    if constexpr (std::is_same_v<E1, glm::vec3> && std::is_same_v<E2, glm::vec3>) {
                        return glm::cross(v1[0], v2[0]);
                    }
                    else {
                        throw makeError<UnimplError>("unsupport type for `cross`");
                    }
                    }, args[0].value, args[1].value);

                ZfxVariable var;
                var.value = std::vector<glm::vec3>{ res };
                return var;
            }
            if (funcname == "floor") {
                const ZfxVariable& var = args[0];
                return call_unary_numeric_func<float>([](float val)->float {return std::floor(val); }, args[0], filter);
            }
            if (funcname == "min") {
                const ZfxVariable& var = args[0];
                //TODO: 目前只考虑一个数值的min
                float cmpval = get_zfxvec_front_elem<float>(args[1].value);
                return call_unary_numeric_func<float>([&](float val)->float {
                    return zeno::min(cmpval, std::abs(val)); }, args[0], filter);
            }
            if (funcname == "max" || funcname == "min") {
                const ZfxVariable& left = args[0];
                const ZfxVariable& right = args[1];
                const int Nleft = left.size();
                const int Nright = right.size();
                const int N = std::max(Nleft, Nright);

                ZfxVariable ret;
                bool bMax = funcname == "max";

                std::visit([&](const auto& lvec, const auto& rvec) {
                    using T1 = std::decay_t<decltype(lvec)>;
                    using E1 = typename T1::value_type;
                    using T2 = std::decay_t<decltype(rvec)>;
                    using E2 = typename T2::value_type;

                    constexpr bool are_int_or_float_v =
                        (std::is_same_v<E1, int> || std::is_same_v<E1, float>) &&
                        (std::is_same_v<E2, int> || std::is_same_v<E2, float>);

                    if constexpr (are_int_or_float_v) {
                        std::vector<float> retvec(N);
                        for (int i = 0; i < N; i++) {
                            auto _arg1 = lvec[std::min(i, Nleft - 1)];
                            auto _arg2 = rvec[std::min(i, Nright - 1)];
                            if (bMax) {
                                retvec[i] = std::max((float)_arg1, (float)_arg2);
                            }
                            else {
                                retvec[i] = std::min((float)_arg1, (float)_arg2);
                            }
                        }
                        ret.value = std::move(retvec);
                    }
                    else {
                        throw makeError<UnimplError>("not support type");
                    }
                    }, left.value, right.value);

                return ret;
            }
            if (funcname == "clamp") {
                const ZfxVariable& var = args[0];
                const ZfxVariable& arg_min = args[1], arg_max = args[2];
                float min = 0, max = 0;
                min = get_zfxvec_front_elem<float>(arg_min.value);
                max = get_zfxvec_front_elem<float>(arg_max.value);

                return call_unary_numeric_func<float>([&](float val)->float {
                    return std::min(std::max(val, min), max);
                    }, args[0], filter);
            }
            if (funcname == "setud") {
                zeno::String key = zeno::stdString2zs(get_zfxvec_front_elem<std::string>(args[0].value));
                auto ud = pContext->spObject->userData();

                std::visit([&](auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    if constexpr (std::is_same_v<E, int>) {
                        ud->set_int(key, vec[0]);
                    }
                    else if constexpr (std::is_same_v<E, float>) {
                        ud->set_float(key, vec[0]);
                    }
                    else if constexpr (std::is_same_v<E, std::string>) {
                        ud->set_string(key, stdString2zs(vec[0]));
                    }
                    else if constexpr (std::is_same_v<E, glm::vec2>) {
                        ud->set_vec2f(key, Vec2f(vec[0][0], vec[0][1]));
                    }
                    else if constexpr (std::is_same_v<E, glm::vec3>) {
                        ud->set_vec3f(key, Vec3f(vec[0][0], vec[0][1], vec[0][2]));
                    }
                    else {
                        throw makeError<UnimplError>("not support type");
                    }
                    }, args[1].value);

                ZfxVariable ret;
                return ret;
            }
            if (funcname == "prim_has_attr") {
                const std::string& nodename = get_zfxvec_front_elem<std::string>(args[0].value);
                const std::string& attrname = get_zfxvec_front_elem<std::string>(args[1].value);

                NodeImpl* pObjNode = pContext->spNode->getGraph()->getNode(nodename);
                if (!pObjNode) {
                    throw makeError<UnimplError>("unknown node `" + nodename + "`");
                }
                auto targetObj = pObjNode->get_default_output_object();
                if (!targetObj)
                    throw makeError<UnimplError>("get nullptr obj from default output when `prim_has_attr` is called");

                if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(targetObj)) {
                    zfxvariant ret = (int)spGeo->has_point_attr(stdString2zs(attrname));
                    return initVarFromZvar(ret);
                }
                else {
                    throw makeError<UnimplError>("get not geometry obj when calling `prim_has_attr`");
                }
            }
            if (funcname == "getud") {
                const std::string& nodename = get_zfxvec_front_elem<std::string>(args[0].value);
                const std::string& objparam = get_zfxvec_front_elem<std::string>(args[1].value);
                const std::string& key = get_zfxvec_front_elem<std::string>(args[2].value);

                NodeImpl* pObjNode = pContext->spNode->getGraph()->getNode(nodename);
                if (!pObjNode) {
                    throw makeError<UnimplError>("unknown node `" + nodename + "`");
                }
                auto targetObj = pObjNode->get_output_obj(objparam);
                if (!targetObj)
                    throw makeError<UnimplError>("get nullptr obj from `" + objparam + "` when `getud` is called");

                auto ud = targetObj->userData();
                zeno::UserData* pUserData = static_cast<zeno::UserData*>(ud);

                ZfxVariable ret;
                auto spud = pUserData->m_data[key].get();
                if (auto strobj = dynamic_cast<zeno::StringObject*>(spud)) {
                    ret = initVarFromZvar(strobj->get());
                }
                else if (auto numobj = dynamic_cast<zeno::NumericObject*>(spud)) {
                    NumericValue val = numobj->value;
                    std::visit([&](auto&& arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int>) {
                            ret = initVarFromZvar(arg);
                        }
                        else if constexpr (std::is_same_v<T, float>) {
                            ret = initVarFromZvar(arg);
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec2i>) {
                            ret = initVarFromZvar(glm::vec2(arg[0], arg[1]));
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec2f>) {
                            ret = initVarFromZvar(glm::vec2(arg[0], arg[1]));
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec3i>) {
                            ret = initVarFromZvar(glm::vec3(arg[0], arg[1], arg[2]));
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec3f>) {
                            ret = initVarFromZvar(glm::vec3(arg[0], arg[1], arg[2]));
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec4i>) {
                            ret = initVarFromZvar(glm::vec4(arg[0], arg[1], arg[2], arg[3]));
                        }
                        else if constexpr (std::is_same_v<T, zeno::vec4f>) {
                            ret = initVarFromZvar(glm::vec4(arg[0], arg[1], arg[2], arg[3]));
                        }
                        }, val);
                }
                else {
                    throw makeError<UnimplError>("unknown type from userdata");
                }
                return ret;
            }
            throw makeError<UnimplError>("unknown function call `" + funcname + "`");
        }
    }
}