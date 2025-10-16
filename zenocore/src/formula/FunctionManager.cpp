#include <zeno/core/FunctionManager.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/Error.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/utils/helper.h>
#include <variant>
#include <functional>
#include <zeno/utils/format.h>
#include <numeric>
#include <zeno/geo/geometryutil.h>
#include <zeno/types/GeometryObject.h>
#include <zeno/utils/vectorutil.h>
#include <reflect/type.hpp>
#include "../utils/zfxutil.h"
#include "functionimpl.h"
#include "funcDesc.h"


using namespace zeno::types;
using namespace zeno::reflect;
using namespace zeno::zfx;

namespace zeno {

    const std::regex FunctionManager::refPattern(R"([\.]?(\/\s*[a-zA-Z0-9\.]+\s*)+)");
    const std::regex FunctionManager::refStrPattern(R"(.*"[\.]?(\/\s*[a-zA-Z0-9\.]+\s*)+".*)");

    FunctionManager::FunctionManager() {
    }

    std::vector<std::string> FunctionManager::getCandidates(const std::string& prefix, bool bFunc) const {
        std::vector<std::string> candidates;
        if (bFunc && prefix.empty())
            return candidates;

        if (bFunc) {
            for (auto& [k, v] : funcsDesc) {
                //TODO: optimize the search
                if (k.substr(0, prefix.size()) == prefix) {
                    candidates.push_back(k);
                }
            }
        }
        else {
            static std::vector<std::string> vars = { "F", "FPS", "T", "PI" };
            for (auto& var : vars) {
                if (var.substr(0, prefix.size()) == prefix) {
                    candidates.push_back(var);
                }
            }
        }
        return candidates;
    }

    std::string FunctionManager::getFuncTip(const std::string& funcName, bool& bExist) const {
        auto iter = funcsDesc.find(funcName);
        if (iter == funcsDesc.end()) {
            bExist = false;
            return "";
        }
        bExist = true;
        return iter->second.tip;
    }

    ZENO_API FUNC_INFO FunctionManager::getFuncInfo(const std::string& funcName) const {
        auto iter = funcsDesc.find(funcName);
        if (iter == funcsDesc.end()) {
            return FUNC_INFO();
        }
        return iter->second;
    }

    static int getElementCount(IObject* spObject, GeoAttrGroup runover) {
        switch (runover)
        {
        case ATTR_POINT: {
            if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(spObject)) {
                return spGeo->npoints();
            }
            else if (auto spPrim = dynamic_cast<PrimitiveObject*>(spObject)) {
                return spPrim->verts->size();
            }
            else {
                return 0;
            }
        }
        case ATTR_FACE:
            if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(spObject)) {
                return spGeo->nfaces();
            }
            else if (auto spPrim = dynamic_cast<PrimitiveObject*>(spObject)) {
                if (spPrim->tris.size() > 0)
                    return spPrim->tris.size();
                else
                    return spPrim->polys.size();
            }
            else {
                return 0;
            }
        case ATTR_GEO: {
            //only one element
            return 1;
        }
        default:
            return 0;
        }
    }

    void FunctionManager::executeZfx(std::shared_ptr<ZfxASTNode> root, ZfxContext* pCtx) {
        //printSyntaxTree(root, pCtx->code);
        auto spGeom = static_cast<GeometryObject_Adapter*>(pCtx->spObject.get());
        bool bConvertHalfEdge = false;
        //不需要转换了，后续IndiceMeshes也得实现同样的几何api，会出现两者混用的情况
        /*
        * //检查语法树，观察是否存在几何拓扑的增删改，如有则转为半边结构的拓扑
        if (spGeom && spGeom->m_impl->type() == zeno::Topo_IndiceMesh) {
            bConvertHalfEdge = hasGeomTopoQueryModify(root);
            if (bConvertHalfEdge) {
                pCtx->spObject = spGeom->toHalfEdgeTopo();
                spGeom = static_cast<GeometryObject_Adapter*>(pCtx->spObject.get());
            }
        }
        */

        pCtx->zfxVariableTbl = &m_globalAttrCached;
        if (pCtx->spObject) {
            int nFilterSize = getElementCount(pCtx->spObject.get(), pCtx->runover);
            ZfxElemFilter filter(nFilterSize, 1);
            scope_exit sp([&] {m_globalAttrCached.clear(); });
            execute(root, filter, pCtx);
        }
        else if (!pCtx->param_constrain.constrain_param.empty()) {
            ZfxElemFilter filter(1, 1);
            execute(root, filter, pCtx);
        }
        else {
            throw makeError<UnimplError>("no object or param constrain when executing zfx");
        }

        if (bConvertHalfEdge && spGeom) {
            pCtx->spObject = spGeom->toIndiceMeshesTopo();
        }
    }

    static ZfxVector anyToZfxVector(Any const& var) {
        if (!var.has_value())
            return ZfxVector();
        if (get_type<bool>() == var.type()) {
            int res = any_cast<bool>(var);
            return std::vector<int>{ res };
        }
        else if (get_type<int>() == var.type()) {
            return std::vector<int>{ any_cast<int>(var) };
        }
        else if (get_type<float>() == var.type()) {
            return std::vector<float>{ any_cast<float>(var) };
        }
        else if (get_type<std::string>() == var.type()) {
            return std::vector<std::string>{ zeno::any_cast_to_string(var) };
        }
        else if (get_type<const char*>() == var.type()) {
            return std::vector<std::string>{ zeno::any_cast_to_string(var) };
        }
        else if (get_type<zeno::vec2i>() == var.type()) {
            auto vec = any_cast<zeno::vec2i>(var);
            return std::vector{ glm::vec2{ vec[0], vec[1] } };
        }
        else if (get_type<zeno::vec3i>() == var.type()) {
            auto vec = any_cast<zeno::vec3i>(var);
            return std::vector{ glm::vec3{ vec[0], vec[1], vec[2] } };
        }
        else if (get_type<zeno::vec4i>() == var.type()) {
            auto vec = any_cast<zeno::vec4i>(var);
            return std::vector{ glm::vec4{ vec[0], vec[1], vec[2], vec[3] } };
        }
        else if (get_type<zeno::vec2f>() == var.type()) {
            auto vec = any_cast<zeno::vec2f>(var);
            return std::vector{ glm::vec2{ vec[0], vec[1] } };
        }
        else if (get_type<zeno::vec3f>() == var.type()) {
            auto vec = any_cast<zeno::vec3f>(var);
            return std::vector{ glm::vec3{ vec[0], vec[1], vec[2] } };
        }
        else if (get_type<zeno::vec4f>() == var.type()) {
            auto vec = any_cast<zeno::vec4f>(var);
            return std::vector{ glm::vec4{ vec[0], vec[1], vec[2], vec[3] } };
        }
        else if (get_type<zeno::PrimVar>() == var.type()) {
            zeno::PrimVar primvar = any_cast<zeno::PrimVar>(var);
            return std::visit([](zeno::PrimVar&& pvar) -> ZfxVector {
                using T = std::decay_t<decltype(pvar)>;
                if constexpr (std::is_same_v<int, T>) {
                    return std::vector<int>{ std::get<int>(pvar) };
                }
                else if constexpr (std::is_same_v<float, T>) {
                    return std::vector<float>{ std::get<float>(pvar) };
                }
                else if constexpr (std::is_same_v<std::string, T>) {
                    return std::vector<std::string>{ std::get<std::string>(pvar) };
                }
                else {
                    return ZfxVector();
                }
                }, primvar);
        }
        else {
            assert(false);
            return ZfxVector();
        }
    }

    static zfxvariant anyToZfxVariant(Any const& var) {
        if (!var.has_value())
            return zfxvariant();
        if (get_type<bool>() == var.type()) {
            int res = any_cast<bool>(var);
            return res;
        }
        else if (get_type<int>() == var.type()) {
            return any_cast<int>(var);
        }
        else if (get_type<float>() == var.type()) {
            return any_cast<float>(var);
        }
        else if (get_type<std::string>() == var.type()) {
            return zeno::any_cast_to_string(var);
        }
        else if (get_type<const char*>() == var.type()) {
            return zeno::any_cast_to_string(var);
        }
        else if (get_type<zeno::vec2i>() == var.type()) {
            auto vec = any_cast<zeno::vec2i>(var);
            return glm::vec2{ vec[0], vec[1] };
        }
        else if (get_type<zeno::vec3i>() == var.type()) {
            auto vec = any_cast<zeno::vec3i>(var);
            return glm::vec3{ vec[0], vec[1], vec[2] };
        }
        else if (get_type<zeno::vec4i>() == var.type()) {
            auto vec = any_cast<zeno::vec4i>(var);
            return glm::vec4{ vec[0], vec[1], vec[2], vec[3] };
        }
        else if (get_type<zeno::vec2f>() == var.type()) {
            auto vec = any_cast<zeno::vec2f>(var);
            return glm::vec2{ vec[0], vec[1] };
        }
        else if (get_type<zeno::vec3f>() == var.type()) {
            auto vec = any_cast<zeno::vec3f>(var);
            return glm::vec3{ vec[0], vec[1], vec[2] };
        }
        else if (get_type<zeno::vec4f>() == var.type()) {
            auto vec = any_cast<zeno::vec4f>(var);
            return glm::vec4{ vec[0], vec[1], vec[2], vec[3] };
        }
        else if (get_type<zeno::PrimVar>() == var.type()) {
            zeno::PrimVar primvar = any_cast<zeno::PrimVar>(var);
            return std::visit([](zeno::PrimVar&& pvar) -> zfxvariant {
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
                    return zfxvariant();
                }
                }, primvar);
        }
        else {
            assert(false);
            return zfxvariant();
        }
    }

    template <class T>
    static T get_zfxvar(zfxvariant value) {
        return std::visit([](auto const& val) -> T {
            using V = std::decay_t<decltype(val)>;
            if constexpr (!std::is_constructible_v<T, V>) {
                if constexpr (std::is_same_v<T, glm::vec3> && std::is_same_v<V, zfxfloatarr>) {
                    return glm::vec3(val[0], val[1], val[2]);
                }
                throw makeError<TypeError>(typeid(T), typeid(V), "get<zfxvariant>");
            }
            else {
                return T(val);
            }
        }, value);
    }

    template <class _Ty = void>
    struct glmdot {
        using _FIRST_ARGUMENT_TYPE_NAME _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS = _Ty;
        using _SECOND_ARGUMENT_TYPE_NAME _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS = _Ty;
        using _RESULT_TYPE_NAME _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS = _Ty;

        _NODISCARD constexpr float operator()(const _Ty& _Left, const _Ty& _Right) const {
            return glm::dot(_Left, _Right);
        }
    };

    template <>
    struct glmdot<void> {
        template <class _Ty1, class _Ty2>
        _NODISCARD constexpr float operator()(_Ty1&& _Left, _Ty2&& _Right) const {
            return glm::dot(static_cast<_Ty1&&>(_Left), static_cast<_Ty2&&>(_Right));
        }
    };


    template<typename Operator>
    ZfxVariable calc_exp(const ZfxVariable& lhs, const ZfxVariable& rhs, const ZfxElemFilter& filter, Operator method) {
        int N1 = lhs.size();
        int N2 = rhs.size();
        int minsize = min(N1, N2);
        int maxsize = max(N1, N2);
        if (N1 != N2) {
            if (minsize != 1)
                throw makeError<UnimplError>("size invalidation on calc_exp");
        }

        ZfxVariable res;

        std::visit([&](const auto& lhs_vec, const auto& rhs_vec) {
            using T1 = std::decay_t<decltype(lhs_vec)>;
            using E1 = typename T1::value_type;
            using T2 = std::decay_t<decltype(rhs_vec)>;
            using E2 = typename T2::value_type;
            using Op = std::decay_t<decltype(method)>;

            if constexpr (std::is_same_v<Op, std::modulus<>>) {
                std::vector<int> result(maxsize);
                for (int i = 0; i < maxsize; i++) {
                    if (!filter[i])
                        continue;
                    if constexpr ((std::is_same_v<E1, int> && std::is_same_v<E2, int>) ||
                            (std::is_same_v<E1, int> && std::is_same_v<E2, float>) ||
                            (std::is_same_v<E1, float> && std::is_same_v<E2, int>) ||
                            (std::is_same_v<E1, float> && std::is_same_v<E2, float>))
                    {
                        int lval = lhs_vec[std::min(i, N1-1)];
                        int rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, rval);
                    }
                    else {
                        throw makeError<UnimplError>("error type for mudulus");
                    }
                }
                res.value = result;
            }
            else if constexpr (std::is_same_v<Op, glmdot<>>)
            {
                if constexpr (std::is_same_v<E1, E2> &&
                    (std::is_same_v<E1, glm::vec2> ||
                        std::is_same_v<E1, glm::vec3> ||
                        std::is_same_v<E1, glm::vec4>))
                {
                    std::vector<float> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, rval);
                    }
                    res.value = result;
                }
                else {
                    throw makeError<UnimplError>("error type for glmdot");
                }
            }
            else if constexpr (
                (std::is_same_v<E1, int> || std::is_same_v<E1, float>) &&
                (std::is_same_v<E2, int> || std::is_same_v<E2, float>))
            {
                if constexpr (std::is_same_v<E1, E2> && std::is_same_v<E1, int>) {
                    std::vector<E1> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        int lval = lhs_vec[std::min(i, N1-1)];
                        int rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, rval);
                    }
                    res.value = result;
                }
                else {
                    //意味着bool运算也可能会用float
                    std::vector<float> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        float lval = lhs_vec[std::min(i, N1-1)];
                        float rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, rval);
                    }
                    res.value = result;
                }
            }
            else if constexpr (
                std::is_same_v<E1, E2> &&
                (std::is_same_v<E1, glm::vec2> ||
                 std::is_same_v<E1, glm::vec3> ||
                 std::is_same_v<E2, glm::vec4>))
            {
                if constexpr (std::is_same_v<Op, std::less_equal<>> ||
                    std::is_same_v<Op, std::less<>> ||
                    std::is_same_v<Op, std::greater<>> ||
                    std::is_same_v<Op, std::greater_equal<>> ||
                    std::is_same_v<Op, std::logical_or<>> || 
                    std::is_same_v<Op, std::logical_and<>> ||
                    std::is_same_v<Op, std::equal_to<>> ||
                    std::is_same_v<Op, std::not_equal_to<>>)
                {
                    throw makeError<UnimplError>("not support boolean exp for glm::vec");
                }
                else
                {
                    std::vector<E1> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, rval);
                    }
                    res.value = result;
                }
            }
            else if constexpr (
                std::is_same_v<E1, glm::vec2> && (std::is_same_v<int, E2> || std::is_same_v<float, E2>) ||
                std::is_same_v<E1, glm::vec3> && (std::is_same_v<int, E2> || std::is_same_v<float, E2>) ||
                std::is_same_v<E2, glm::vec4> && (std::is_same_v<int, E2> || std::is_same_v<float, E2>))
            {
                if constexpr (std::is_same_v<Op, std::less_equal<>> ||
                    std::is_same_v<Op, std::less<>> ||
                    std::is_same_v<Op, std::greater<>> ||
                    std::is_same_v<Op, std::greater_equal<>> ||
                    std::is_same_v<Op, std::logical_or<>> ||
                    std::is_same_v<Op, std::logical_and<>> ||
                    std::is_same_v<Op, std::equal_to<>> ||
                    std::is_same_v<Op, std::not_equal_to<>>) {
                    throw makeError<UnimplError>("");
                }
                else
                {
                    std::vector<E1> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, E1(rval));
                    }
                    res.value = result;
                }
            }
            else if constexpr (
                (std::is_same_v<int, E1> || std::is_same_v<float, E1>) && std::is_same_v<E2, glm::vec2> ||
                (std::is_same_v<int, E1> || std::is_same_v<float, E1>) && std::is_same_v<E2, glm::vec3> ||
                (std::is_same_v<int, E1> || std::is_same_v<float, E1>) && std::is_same_v<E2, glm::vec4>)
            {
                if constexpr (std::is_same_v<Op, std::less_equal<>> ||
                    std::is_same_v<Op, std::less<>> ||
                    std::is_same_v<Op, std::greater<>> ||
                    std::is_same_v<Op, std::greater_equal<>> ||
                    std::is_same_v<Op, std::logical_or<>> ||
                    std::is_same_v<Op, std::logical_and<>> ||
                    std::is_same_v<Op, std::equal_to<>> ||
                    std::is_same_v<Op, std::not_equal_to<>>) {
                    throw makeError<UnimplError>("");
                }
                else
                {
                    std::vector<E2> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(E2(lval), rval);
                    }
                    res.value = result;
                }
            }
            else if constexpr (
                std::is_same_v<E1, glm::mat3> && std::is_same_v<E1, E2> ||
                std::is_same_v<E1, glm::mat4> && std::is_same_v<E1, E2> ||
                std::is_same_v<E1, glm::mat2> && std::is_same_v<E1, E2>)
            {
                if constexpr (
                    std::is_same_v<Op, std::less_equal<>> ||
                    std::is_same_v<Op, std::less<>> ||
                    std::is_same_v<Op, std::greater_equal<>> ||
                    std::is_same_v<Op, std::greater<>> ||
                    std::is_same_v<Op, std::logical_or<>> ||
                    std::is_same_v<Op, std::logical_and<>> ||
                    std::is_same_v<Op, std::equal_to<>> ||
                    std::is_same_v<Op, std::not_equal_to<>>) {
                    throw makeError<UnimplError>("");
                }
                else if constexpr (std::is_same_v<Op, std::multiplies<>>)
                {
                    //glm的实现里，乘法是顺序相反的，比如A*B, 其实是我们理解的B * A.
                    std::vector<E1> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(rval, lval);
                    }
                    res.value = result;
                }
                else {
                    std::vector<E1> result(maxsize);
                    for (int i = 0; i < maxsize; i++) {
                        if (!filter[i])
                            continue;
                        const auto& lval = lhs_vec[std::min(i, N1-1)];
                        const auto& rval = rhs_vec[std::min(i, N2-1)];
                        result[i] = method(lval, E1(rval));
                    }
                    res.value = result;
                }
            }
            else if constexpr (
                (std::is_same_v<Op, std::equal_to<>> || std::is_same_v<Op, std::not_equal_to<>>) &&
                std::is_same_v<E1, std::string> && std::is_same_v<E1, E2>)
            {
                std::vector<int> result(maxsize);
                for (int i = 0; i < maxsize; i++) {
                    if (!filter[i])
                        continue;
                    const auto& lval = lhs_vec[std::min(i, N1-1)];
                    const auto& rval = rhs_vec[std::min(i, N2-1)];
                    result[i] = method(lval, E1(rval));
                }
                res.value = result;
            }
            else {
                throw makeError<UnimplError>("unknown op or data type when executing `call_exp`");
            }
            }, lhs.value, rhs.value);
        return res;
    }

    void FunctionManager::testExp() {

        //glm::vec3 v1(1, 2, 3);
        //glm::vec3 v2(1, 3, 1);

        //zfxvariant vec1 = v1;
        //zfxvariant vec2 = v2;
        //zfxvariant vec3 = calc_exp(v1, v2, std::divides());

        //glm::mat3 mat1 = glm::mat3({ {1., 0, 2.}, {2., 1., -1.}, {0, 1., 1.} });
        //glm::mat3 mat2 = glm::mat3({ {1., 0, 0}, {0, -1., 1.}, {0, 0, -1.} });
        //glm::mat3 mat3 = glm::mat3({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });
        //glm::mat3 mat4 = glm::mat3({ {1, 0, 0}, {0, -1, 1}, {0, 0, -1} });
        //glm::mat3 mm = mat3 * mat4;
        //mm = mat1 * mat2;

        //zfxvariant bval = calc_exp(mat1, mat2, std::equal_to());
        //zfxvariant mmm = calc_exp(mat1, mat2, std::multiplies());

        //glm::mat3 mm2 = glm::dot(mat1, mat2);
    }

    static void set_array_element(ZfxVariable& zfxarr, ZfxVariable idxarr, const ZfxVariable& zfxvalue) {
        std::visit([&](auto& target_vec, const auto& idx_vec, const auto& value_vec) {
            using T = std::decay_t<decltype(target_vec)>;
            using T_ElementType = typename T::value_type;
            using E = std::decay_t<decltype(value_vec)>;
            using E_ElementType = typename E::value_type;
            using IDX = std::decay_t<decltype(idx_vec)>;
            using IDX_TYPE = typename IDX::value_type;

            if constexpr (std::is_same_v<IDX_TYPE, int> || std::is_same_v<IDX_TYPE, float>) {
                //检查索引大小合法性
                int idx = idx_vec.size() == 1 ? idx_vec[0] : -1;

                if constexpr ((std::is_same_v<T_ElementType, float> ||
                    std::is_same_v<T_ElementType, int>) &&
                    (std::is_same_v<E_ElementType, float> ||
                        std::is_same_v<E_ElementType, int>))
                {
                    int N = target_vec.size();
                    if (idx_vec.size() == 1) {
                        target_vec[idx_vec[0]] = value_vec[idx_vec[0]];
                    }
                    else {
                        for (int i = 0; i < N; i++) {
                            int idx = idx_vec[i];
                            target_vec[idx] = value_vec[idx];
                        }
                    }
                }
                else if constexpr (std::is_same_v<T_ElementType, E_ElementType>)
                {
                    int N = target_vec.size();
                    if (idx_vec.size() == 1) {
                        target_vec[idx_vec[0]] = value_vec[idx_vec[0]];
                    }
                    else {
                        for (int i = 0; i < N; i++) {
                            int idx = idx_vec[i];
                            target_vec[idx] = value_vec[idx];
                        }
                    }
                }
                else {
                    throw makeError<UnimplError>("unsupport type to call `set_array_element`");
                }
            }
            else {
                throw makeError<UnimplError>("[zfx::set_array_element]idx type should be `int` or `float`");
            }
        }, zfxarr.value, idxarr.value, zfxvalue.value);
    }

    static ZfxVariable get_array_element(const ZfxVariable& arr, const ZfxVariable& varidx) {
        //取的元素是指一个数据的分量，而不是某一个位元（点线面）的数据
        return std::visit([&](const auto& vec, const auto& idxvec)->ZfxVariable {
            using T = std::decay_t<decltype(vec)>;
            using E = typename T::value_type;
            using IDX = std::decay_t<decltype(idxvec)>;
            using IDX_TYPE = typename IDX::value_type;

            if constexpr (std::is_same_v<IDX_TYPE, int> || std::is_same_v<IDX_TYPE, float>) {
                if constexpr (std::is_same_v<E, zfxintarr> ||
                    std::is_same_v<E, zfxfloatarr> ||
                    //std::is_same_v<E, zfxstringarr> ||
                    //std::is_same_v<E, zfxvec2arr> ||
                    //std::is_same_v<E, zfxvec3arr> ||
                    //std::is_same_v<E, zfxvec4arr> ||
                    std::is_same_v<E, glm::vec2> ||
                    std::is_same_v<E, glm::vec3> ||
                    std::is_same_v<E, glm::vec4>/* ||
                    std::is_same_v<E, glm::mat2> ||
                    std::is_same_v<E, glm::mat3> ||
                    std::is_same_v<E, glm::mat4>*/) {

                    ZfxVariable res;
                    std::vector<float> ret_vec(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        int idx = (idxvec.size() == 1) ? idxvec[0] : idxvec[i];
                        ret_vec[i] = vec[i][idx];
                    }
                    res.value = std::move(ret_vec);
                    return res;
                }
                else {
                    throw makeError<UnimplError>("no support type to call `get_array_element`");
                }
            }
            else {
                throw makeError<UnimplError>("[zfx::get_array_element]idx type should be `int` or `float`");
            }
            }, arr.value, varidx.value);
    }

    static ZfxVariable get_element_by_name(const ZfxVariable& arr, const std::string& name) {
        int idx = -1;
        if (name == "x") idx = 0;
        else if (name == "y") idx = 1;
        else if (name == "z") idx = 2;
        else if (name == "w") idx = 3;
        else
        {
            throw makeError<UnimplError>("Indexing Exceed");
        }
        ZfxVariable varidx;
        varidx.value = std::vector{ idx };
        return get_array_element(arr, varidx);
    }

    static void set_element_by_name(ZfxVariable& arr, const std::string& name, const ZfxVariable& value) {
        int idx = -1;
        if (name == "x") idx = 0;
        else if (name == "y") idx = 1;
        else if (name == "z") idx = 2;
        else if (name == "w") idx = 3;
        else throw makeError<UnimplError>("index error.");
        ZfxVariable varidx;
        varidx.value = std::vector{ idx };
        set_array_element(arr, varidx, value);
    }

    static void selfIncOrDec(ZfxVariable& var, bool bInc) {
        std::visit([&](auto& vec) {
            using T = std::decay_t<decltype(vec)>;
            using E = typename T::value_type;
            if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                for (auto& elem : vec) {
                    elem++;
                }
            }
            else {
                throw makeError<UnimplError>("only support `int` and `float` to self inc or dec");
            }
            }, var.value);
    }

    std::vector<ZfxVariable> FunctionManager::process_args(std::shared_ptr<ZfxASTNode> parent, ZfxElemFilter& filter, ZfxContext* pContext) {
        std::vector<ZfxVariable> args;
        for (auto pChild : parent->children) {
            ZfxVariable argval = execute(pChild, filter, pContext);
            args.emplace_back(argval);
        }
        return args;
    }

    void FunctionManager::pushStack() {
        m_stacks.push_back(ZfxStackEnv());
    }

    void FunctionManager::popStack() {
        m_stacks.pop_back();
    }

    ZfxVariable& FunctionManager::getVariableRef(const std::string& name, ZfxContext* pContext) {
        assert(!name.empty());
        if (name.at(0) != '@'){
            for (auto iter = m_stacks.rbegin(); iter != m_stacks.rend(); iter++) {
                auto& stackvars = iter->table;
                auto iter_ = stackvars.find(name);
                if (iter_ != stackvars.end()) {
                    return stackvars.at(name);
                }
            }
        }
        throw makeError<KeyError>(name, "variable `" + name + "` not founded");
    }

    bool FunctionManager::declareVariable(const std::string& name) {
        if (m_stacks.empty()) {
            return false;
        }
        auto iterCurrentStack = m_stacks.rbegin();
        auto& vars = iterCurrentStack->table;
        if (vars.find(name) != vars.end()) {
            return false;
        }
        ZfxVariable variable;
        vars.insert(std::make_pair(name, variable));
        return true;
    }

    bool FunctionManager::assignVariable(const std::string& name, ZfxVariable var, ZfxContext* pContext) {
        if (m_stacks.empty()) {
            return false;
        }
        ZfxVariable& self = getVariableRef(name, pContext);
        self = var;
        return true;
    }

    void FunctionManager::validateVar(operatorVals vartype, ZfxVariable& vars) {
        std::visit([&](auto& vec) {
            using T = std::decay_t<decltype(vec)>;
            using E = typename T::value_type;
            if constexpr (std::is_same_v<E, float>) {
                switch (vartype) {
                case TYPE_INT: {
                    std::vector<int> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        newvars[i] = vec[i];
                    }
                    vars.value = newvars;
                    break;
                }
                case TYPE_FLOAT: break;
                default: {
                    throw makeError<UnimplError>("type dismatch `float`");
                }
                }
            }
            else if constexpr (std::is_same_v<E, int>) {
                switch (vartype) {
                case TYPE_FLOAT: {
                    std::vector<float> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++)
                        newvars[i] = vec[i];
                    vars.value = newvars;
                    break;
                }
                case TYPE_INT: break;
                default: {
                    throw makeError<UnimplError>("type dismatch `int`");
                }
                }
            }
            else if constexpr (std::is_same_v<E, zfxintarr>) {
                switch (vartype) {
                case TYPE_FLOAT_ARR: {
                    std::vector<zfxfloatarr> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        newvars[i].resize(vec[i].size());
                        for (int j = 0; j < newvars[i].size(); j++) {
                            newvars[i][j] = vec[i][j];
                        }
                    }
                    vars.value = newvars;
                    break;
                }
                case TYPE_INT_ARR: break;
                default: throw makeError<UnimplError>("type dismatch `zfxintarr`");
                }
            }
            else if constexpr (std::is_same_v<E, zfxfloatarr>) {
                switch (vartype) {
                case TYPE_INT_ARR: {
                    std::vector<zfxintarr> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        newvars[i].resize(vec[i].size());
                        for (int j = 0; j < newvars[i].size(); j++) {
                            newvars[i][j] = vec[i][j];
                        }
                    }
                    vars.value = newvars;
                    break;
                }
                case TYPE_VECTOR2: {
                    std::vector<glm::vec2> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        if (vec[i].size() != 2) throw makeError<UnimplError>("num of elements of arr dismatch");
                        newvars[i] = { vec[i][0], vec[i][1] };
                    }
                    break;
                }
                case TYPE_VECTOR3: {
                    std::vector<glm::vec3> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        if (vec[i].size() != 3) throw makeError<UnimplError>("num of elements of arr dismatch");
                        newvars[i] = { vec[i][0], vec[i][1], vec[i][2] };
                    }
                    break;
                }
                case TYPE_VECTOR4: {
                    std::vector<glm::vec4> newvars(vec.size());
                    for (int i = 0; i < vec.size(); i++) {
                        if (vec[i].size() != 4) throw makeError<UnimplError>("num of elements of arr dismatch");
                        newvars[i] = { vec[i][0], vec[i][1], vec[i][2], vec[i][3] };
                    }
                    break;
                }
                case TYPE_FLOAT_ARR: break;
                default: throw makeError<UnimplError>("type dismatch `zfxfloatarr`");
                }
            }
            else if constexpr (std::is_same_v<E, std::string>) {
                if (vartype != TYPE_STRING) {
                    throw makeError<UnimplError>("type dismatch TYPE_STRING");
                }
            }
            else if constexpr (std::is_same_v<E, zfxstringarr>) {
                if (vartype != TYPE_STRING_ARR) {
                    throw makeError<UnimplError>("type dismatch TYPE_STRING_ARR");
                }
            }
            else if constexpr (std::is_same_v<E, glm::vec2>) {
                if (vartype != TYPE_VECTOR2) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR2");
                }
            }
            else if constexpr (std::is_same_v<E, glm::vec3>) {
                if (vartype != TYPE_VECTOR3) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR3");
                }
            }
            else if constexpr (std::is_same_v<E, glm::vec4>) {
                if (vartype != TYPE_VECTOR4) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR4");
                }
            }
            else if constexpr (std::is_same_v<E, glm::mat2>) {
                if (vartype != TYPE_MATRIX2) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR2");
                }
            }
            else if constexpr (std::is_same_v<E, glm::mat3>) {
                if (vartype != TYPE_MATRIX3) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR3");
                }
            }
            else if constexpr (std::is_same_v<E, glm::mat4>) {
                if (vartype != TYPE_MATRIX4) {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR4");
                }
            }
            }, vars.value);
    }

    ZfxVariable FunctionManager::parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxElemFilter& filter, ZfxContext* pContext) {
        //目前暂时只考虑 {1,2,3}  {{x1, y1, z1}, {x2,y2,z2}}这种，如果再算上属性值扩展，比较麻烦，而且少见。
        std::vector<ZfxVariable> args = process_args(pNode, filter, pContext);
        //由于不清楚类型，所以先根据值类型构造一个zfxvariant返回出去，再由外部类型定义/赋值来处理
        //由于zfxvariant并没有多维数组，因此遇到多维数组，直接构造矩阵（如果构造失败，那就throw）
        if (args.empty()) {
            //直接返回默认的
            return ZfxVariable();
        }

        zfxfloatarr floatarr;
        zfxstringarr strarr;
        glm::mat2 m2;
        glm::mat3 m3;
        glm::mat4 m4;

        operatorVals dataType = UNDEFINE_OP;
        for (int idx = 0; idx < args.size(); idx++) {
            auto& arg = args[idx];

            std::visit([&](auto& vec) {
                using T = std::decay_t<decltype(vec)>;
                using E = typename T::value_type;
                if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                    if (dataType != TYPE_FLOAT_ARR && dataType != UNDEFINE_OP)
                        throw makeError<UnimplError>("data type inconsistent `float`");
                    dataType = TYPE_FLOAT_ARR;
                    floatarr.push_back(vec[0]);
                }
                else if constexpr (std::is_same_v<E, std::string>) {
                    if (dataType != TYPE_STRING_ARR && dataType != UNDEFINE_OP)
                        throw makeError<UnimplError>("data type inconsistent `string`");
                    dataType = TYPE_STRING_ARR;
                    strarr.push_back(vec[0]);
                }
                else if constexpr (std::is_same_v<E, zfxfloatarr>) {
                    if (dataType != UNDEFINE_OP && dataType != TYPE_MATRIX2 && dataType != TYPE_MATRIX3 && dataType != TYPE_MATRIX4) {
                        throw makeError<UnimplError>("data type inconsistent");
                    }

                    auto& arr = vec[0];
                    if (dataType == UNDEFINE_OP) {
                        if (arr.size() == 2) {
                            dataType = TYPE_MATRIX2;
                        }
                        else if (arr.size() == 3) {
                            dataType = TYPE_MATRIX3;
                        }
                        else if (arr.size() == 4) {
                            dataType = TYPE_MATRIX4;
                        }
                    }

                    //{{0, 1}, {2, 3}}
                    if (dataType == TYPE_MATRIX2 && arr.size() == 2 && idx < 2) {
                        m2[idx][0] = arr[0];
                        m2[idx][1] = arr[1];
                    }
                    //{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
                    else if (dataType == TYPE_MATRIX3 && arr.size() == 3 && idx < 3) {
                        m3[idx][0] = arr[0];
                        m3[idx][1] = arr[1];
                        m3[idx][2] = arr[2];
                    }
                    //{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}}
                    else if (dataType == TYPE_MATRIX4 && arr.size() == 4 && idx < 4) {
                        m4[idx][0] = arr[0];
                        m4[idx][1] = arr[1];
                        m4[idx][2] = arr[2];
                        m4[idx][3] = arr[3];
                    }
                    else {
                        throw makeError<UnimplError>("mat element dims inconsistent");
                    }
                }
                else if constexpr (std::is_same_v<T, zfxintarr>) {
                    //不考虑intarr，因为glm的vector/matrix都是储存float
                    throw makeError<UnimplError>("unsupported type to construct array");
                }
                else {
                    throw makeError<UnimplError>("unsupported type to construct array");
                }
            }, arg.value);
        }

        ZfxVariable res;
        res.bArray = true;
        if (dataType == TYPE_FLOAT_ARR) {
            res.value = std::vector<zfxfloatarr>{ floatarr };
        }
        else if (dataType == TYPE_STRING_ARR) {
            res.value = std::vector<zfxstringarr>{ strarr };
        }
        else if (dataType == TYPE_MATRIX2) {
            res.value = std::vector<glm::mat2>{ m2 };
        }
        else if (dataType == TYPE_MATRIX3) {
            res.value = std::vector<glm::mat3>{ m3 };
        }
        else if (dataType == TYPE_MATRIX4) {
            res.value = std::vector<glm::mat4>{ m4 };
        }
        return res;
    }

    bool FunctionManager::hasGeomTopoQueryModify(std::shared_ptr<ZfxASTNode> pNode) const {
        for (auto child : pNode->children) {
            if (child->type == FUNC) {
                const std::string& funcname = get_zfxvar<std::string>(child->value);
                if (funcname == "add_vertex" ||
                    funcname == "add_point" ||
                    funcname == "add_face" ||
                    funcname == "remove_face" ||
                    funcname == "remove_point" ||
                    funcname == "remove_vertex" ||
                    funcname == "point_faces" ||
                    funcname == "point_vertex" ||
                    funcname == "point_vertices" ||
                    funcname == "face_point" ||
                    funcname == "face_points" ||
                    funcname == "face_vertex" ||
                    funcname == "face_vertex_count" ||
                    funcname == "face_vertices" ||
                    funcname == "vertex_index" ||
                    funcname == "vertex_next" ||
                    funcname == "vertex_prev" ||
                    funcname == "vertex_point" ||
                    funcname == "vertex_face" ||
                    funcname == "vertex_face_index"
                    ) {
                    return true;
                }
            }
            else {
                bool ret = hasGeomTopoQueryModify(child);
                if (ret)
                    return ret;
            }
        }
        return false;
    }

    bool FunctionManager::hasTrue(const ZfxVariable& cond, const ZfxElemFilter& filter, ZfxElemFilter& ifFilter, ZfxElemFilter& elseFilter) const {
        return std::visit([&](auto& vec)->bool {
            using T = std::decay_t<decltype(vec)>;
            using E = typename T::value_type;

            if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                int N = vec.size();
                assert(N == filter.size() || N == 1);
                ifFilter = filter;
                elseFilter = filter;
                bool bret = false;
                for (int i = 0; i < vec.size(); i++) {
                    if (filter[i]) {
                        if (vec[i])
                        {
                            bret = true;
                            elseFilter[i] = 0;
                        }
                        else
                        {
                            ifFilter[i] = 0;
                        }
                    }
                }
                return bret;
            }
            else {
                throw makeError<UnimplError>("not support type");
            }
        }, cond.value);
    }

    static void commitToObject(ZfxContext* pContext, const ZfxVariable& zfxvar, const std::string& attr_name, ZfxElemFilter& filter) {
        assert(!attr_name.empty());
        GeoAttrGroup grp = pContext->runover;
        auto& zfxvec = zfxvar.value;
        if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get())) {
            AttrVar wtf = zeno::zfx::convertToAttrVar(zfxvec);
            std::string attrname;
            if (attr_name[0] == '@')
                attrname = attr_name.substr(1);
            if (!spGeo->m_impl->has_attr(grp, attrname)) {
                spGeo->m_impl->create_attr(grp, attrname, wtf);
            }
            else {
                spGeo->m_impl->set_attr(grp, attrname, wtf);
            }
        }
        else {
            throw makeError<UnimplError>("only support Geometry when setting attributes");
        }
    }

    void FunctionManager::commitToPrim(const std::string& attrname, const ZfxVariable& val, ZfxElemFilter& filter, ZfxContext* pContext) {
        if (pContext->runover == ATTR_POINT) {
            if (attrname == "@P") {
                commitToObject(pContext, val, "pos", filter);
            }
            else if (attrname == "@ptnum") {
                throw makeError<UnimplError>("");
            }
            else if (attrname == "@N") {
                commitToObject(pContext, val, "nrm", filter);
            }
            else if (attrname == "@Cd") {
            }
            else {
                commitToObject(pContext, val, attrname, filter);
            }
        }
        else if (pContext->runover == ATTR_FACE) {
            
        }
        else if (pContext->runover == ATTR_GEO) {
            
        }
        else {
            
        }
    }

    static ZfxVariable getAttrValue_impl(std::shared_ptr<IObject> spObject, const std::string& attr_name) {
        if (auto spPrim = std::dynamic_pointer_cast<PrimitiveObject>(spObject)) {
            if (attr_name == "pos")
            {
                const auto& P = spPrim->attr<vec3f>("pos");
                ZfxVariable res;
                res.bAttr = true;
                std::vector<glm::vec3> vecpos(P.size());
                vecpos.reserve(P.size());
                for (auto pos : P) {
                    vecpos.push_back(glm::vec3(pos[0], pos[1], pos[2]));
                }
                res.value = std::move(vecpos);
                return res;
            }
            if (attr_name == "ptnum")
            {
                int N = spPrim->verts->size();
                ZfxVariable res;
                res.bAttr = true;
                std::vector<int> seq(N);
                seq.reserve(N);
                for (int i = 0; i < N; i++)
                    seq[i] = i;
                res.value = seq;
                return res;
            }
            if (attr_name == "nrm")
            {
                if (spPrim->has_attr("nrm")) {
                    const auto& nrms = spPrim->attr<vec3f>("nrm");
                    std::vector<glm::vec3> _nrms;
                    _nrms.reserve(nrms.size());
                    for (auto nrm : nrms) {
                        _nrms.push_back(glm::vec3(nrm[0], nrm[1], nrm[2]));
                    }
                    ZfxVariable res;
                    res.bAttr = true;
                    res.value = _nrms;
                    return res;
                }
                else {
                    throw makeError<UnimplError>("the prim has no attr about normal, you can check whether the option `hasNormal` is on");
                }
            }
        }
        else if (auto spGeo = dynamic_cast<GeometryObject_Adapter*>(spObject.get())) {
            if (attr_name == "pos")
            {
                const auto& P = spGeo->points_pos();
                ZfxVariable res;
                res.bAttr = true;
                res.value = zvec3toglm(P);
                return res;
            }
            if (attr_name == "ptnum")
            {
                int N = spGeo->npoints();
                ZfxVariable res;
                res.bAttr = true;
                std::vector<int> seq(N);
                seq.reserve(N);
                for (int i = 0; i < N; i++)
                    seq[i] = i;
                res.value = seq;
                return res;
            }
            if (attr_name == "nrm")
            {
                //if (spGeo->has_attr(ATTR_POINT, "nrm")) {
                //    ZfxVariable res;
                //    res.bAttr = true;
                //    res.value = spGeo->get_attr_byzfx(ATTR_POINT, "nrm");
                //    return res;
                //}
                //else {
                //    throw makeError<UnimplError>("the prim has no attr about normal, you can check whether the option `hasNormal` is on");
                //}
            }
        }
        return ZfxVariable();
    }

    void FunctionManager::setAttrValue(const std::string& attrname, const std::string& channel, const ZfxVariable& var, operatorVals opVal, ZfxElemFilter& filter, ZfxContext* pContext) {
        zeno::zfx::setAttrValue(attrname, channel, var, opVal, filter, pContext);
    }

    ZfxVariable FunctionManager::getAttrValue(const std::string& attrname, ZfxContext* pContext, char channel) {
        return zeno::zfx::getAttrValue(attrname, pContext, channel);
    }

    ZfxVariable FunctionManager::trunkVariable(ZfxVariable origin, const ZfxElemFilter& filter) {
        return std::visit([&](auto& vec)->ZfxVariable {
            using T = std::decay_t<decltype(vec)>;
            using E = typename T::value_type;

            int ndim = vec.size();
            int nfilter = filter.size();
            if (nfilter == ndim) {
                return origin;
            }
            else if (nfilter < ndim) {
                //裁剪origin
                ZfxVariable truncate;
                auto trunc_vec = vec;
                trunc_vec.resize(nfilter);
                std::copy(vec.begin(), vec.begin() + nfilter, trunc_vec.begin());
                truncate.value = std::move(vec);
                return truncate;
            }
            else {
                //扩大origin
                vec.resize(nfilter);
                return origin;
            }
            }, origin.value);
    }

    ZfxVariable FunctionManager::execute(std::shared_ptr<ZfxASTNode> root, ZfxElemFilter& filter, ZfxContext* pContext) {
        if (!root) {
            throw makeError<UnimplError>("Indexing Error.");
        }
        switch (root->type)
        {
            case NUMBER:
            case STRING:
            case ATTR_VAR:
            case BOOLTYPE: {
                return initVarFromZvar(root->value);
            }
            case ZENVAR: {
                //这里指的是取zenvar的值用于上层的计算或者输出，赋值并不会走到这里
                const std::string& varname = get_zfxvar<std::string>(root->value);
#if 1
                if (varname == "pos") {
                    int j;
                    j = 0;
                }
#endif
                if (root->bAttr && root->opVal == COMPVISIT) {
                    if (root->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error on NameVisit");
                    }
                    std::string component = get_zfxvar<std::string>(root->children[0]->value);
                    char channel = 0;
                    if (component == "x")
                        channel = 'x';
                    else if (component == "y")
                        channel = 'y';
                    else if (component == "z")
                        channel = 'z';
                    else if (component == "w")
                        channel = 'w';
                    ZfxVariable var = getAttrValue(varname, pContext, channel);
                    return trunkVariable(var, filter);
                }
                else if (root->opVal == Indexing) {

                }

                if (root->bAttr && root->opVal == UNDEFINE_OP) {
                    ZfxVariable var = getAttrValue(varname, pContext);
                    return trunkVariable(var, filter);
                }

                ZfxVariable& var = getVariableRef(varname, pContext);

                switch (root->opVal) {
                case Indexing: {
                    if (root->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error.");
                    }
                    const ZfxVariable& idx = execute(root->children[0], filter, pContext);
                    ZfxVariable elemvar = get_array_element(var, idx);
                    return elemvar;
                }
                case BulitInVar: {
                    std::string attrname = get_zfxvar<std::string>(root->value);
                    if (attrname.size() < 2 || attrname[0] != '$') {
                        throw makeError<UnimplError>("build in var");
                    }
                    attrname = attrname.substr(1);
                    if (attrname == "F") {
                        //TODO
                    }
                    else if (attrname == "T") {
                        //TODO
                    }
                    else if (attrname == "FPS") {
                        //TODO
                    }
                }
                case AutoIncreaseFirst: {
                    selfIncOrDec(var, true);
                    if (root->bAttr)
                        var.bAttrUpdated = true;
                    return var;
                }
                case AutoDecreaseFirst: {
                    selfIncOrDec(var, false);
                    if (root->bAttr)
                        var.bAttrUpdated = true;
                    return var;
                }
                case AutoIncreaseLast:
                case AutoDecreaseLast:   //在外面再自增/减
                {
                    selfIncOrDec(var, AutoIncreaseLast == root->opVal);
                    if (root->bAttr)
                        var.bAttrUpdated = true;
                    return var;
                }
                default: {
                    return var;
                }
                }
            }
            case ASSIGNMENT: {
                //赋值
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("assign variable failed.");
                }

                std::shared_ptr<ZfxASTNode> valNode = root->children[1];
                ZfxVariable res = execute(valNode, filter, pContext);

                std::shared_ptr<ZfxASTNode> zenvarNode = root->children[0];
                if (zenvarNode->bAttr) {
                    //无须把值拎出来再计算，直接往属性数据内部设置
                    std::string attrname = get_zfxvar<std::string>(zenvarNode->value).substr(1);
                    //DEBUG:
#if 0
                    if (attrname == "type") {
                        int j;
                        j = 0;
                    }
#endif
                    if (attrname == "P") {
                        attrname = "pos";
                    }

                    std::string channel;
                    if (zenvarNode->opVal == COMPVISIT) {
                        assert(zenvarNode->children.size() == 1);
                        channel = get_zfxvar<std::string>(zenvarNode->children[0]->value);
                    }
                    else if (zenvarNode->opVal == Indexing) {
                        //todo
                        throw makeError<UnimplError>("Not support indexing for internal attributes");
                    }

                    AttrVar initValue = getInitValueFromVariant(res.value); //拿初值就行
                    auto spGeom = dynamic_cast<GeometryObject_Adapter*>(pContext->spObject.get());
                    if (pContext->runover == ATTR_POINT) {
                        if (!spGeom->m_impl->has_point_attr(attrname)) {
                            spGeom->m_impl->create_point_attr(attrname, initValue);
                        }
                    }
                    else if (pContext->runover == ATTR_VERTEX) {
                        if (!spGeom->m_impl->has_vertex_attr(attrname)) {
                            spGeom->m_impl->create_vertex_attr(attrname, initValue);
                        }
                    }
                    else if (pContext->runover == ATTR_FACE) {
                        if (!spGeom->m_impl->has_face_attr(attrname)) {
                            spGeom->m_impl->create_face_attr(attrname, initValue);
                        }
                    }
                    else if (pContext->runover == ATTR_GEO) {
                        if (!spGeom->m_impl->has_geometry_attr(attrname)) {
                            spGeom->m_impl->create_geometry_attr(attrname, initValue);
                        }
                    }
                    setAttrValue(attrname, channel, res, root->opVal, filter, pContext);
                    return ZfxVariable();
                }

                const std::string& targetvar = get_zfxvar<std::string>(zenvarNode->value);

                if (root->opVal == AddAssign) {
                    ZfxVariable varres = execute(zenvarNode, filter, pContext);
                    res = calc_exp(varres, res, filter, std::plus());
                }
                else if (root->opVal == MulAssign) {
                    ZfxVariable varres = execute(zenvarNode, filter, pContext);
                    res = calc_exp(varres, res, filter, std::multiplies());
                }
                else if (root->opVal == SubAssign) {
                    ZfxVariable varres = execute(zenvarNode, filter, pContext);
                    res = calc_exp(varres, res, filter, std::minus());
                }
                else if (root->opVal == DivAssign) {
                    ZfxVariable varres = execute(zenvarNode, filter, pContext);
                    res = calc_exp(varres, res, filter, std::divides());
                }

                {
                    //能赋值的变量只有：1.普通zfx定义的变量    2.参数约束的参数变量
                    std::string nodeparam = pContext->param_constrain.constrain_param;
                    if (!nodeparam.empty()) {
                        auto spNode = pContext->spNode;
                        bool bInputParam = pContext->param_constrain.bInput;

                        std::visit([&](auto& vec) {
                            using T = std::decay_t<decltype(vec)>;
                            using E = typename T::value_type;
                            if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                                bool bVal = get_zfxvar<int>(vec[0]);
                                if (targetvar == "visible") {
                                    pContext->param_constrain.update_nodeparam_prop = spNode->update_param_visible(nodeparam, bVal, bInputParam);
                                }
                                else if (targetvar == "enabled") {
                                    pContext->param_constrain.update_nodeparam_prop = spNode->update_param_enable(nodeparam, bVal, bInputParam);
                                }
                            }
                            else {
                                throw makeError<UnimplError>("not support type when switch `ASSIGNMENT`");
                            }
                            }, res.value);
                        return ZfxVariable();
                    }

                    //直接解析变量
                    ZfxVariable& var = getVariableRef(targetvar, pContext);

                    if (root->bAttr)
                        var.bAttrUpdated = true;

                    //先赋值结束后直接提交到prim上面（可否在结束的时候提交？好像也行，还能统一操作）
                    scope_exit sp([&]() {
                        if (zenvarNode->bAttr) {
                            commitToPrim(targetvar, var, filter, pContext);
                        }
                    });

                    switch (zenvarNode->opVal) {
                    case Indexing: {
                        if (zenvarNode->children.size() != 1) {
                            throw makeError<UnimplError>("Indexing Error.");
                        }
                        const ZfxVariable& idx = execute(zenvarNode->children[0], filter, pContext);
                        set_array_element(var, idx, res);
                        return ZfxVariable();  //无需返回什么具体的值
                    }
                    case COMPVISIT: {
                        if (zenvarNode->children.size() != 1) {
                            throw makeError<UnimplError>("Indexing Error on NameVisit");
                        }
                        std::string component = get_zfxvar<std::string>(zenvarNode->children[0]->value);
                        set_element_by_name(var, component, res);
                        return ZfxVariable();
                    }
                    case BulitInVar: {
                        //TODO: 什么情况下需要修改这种变量
                        //$F $T这些貌似不能通过脚本去改，houdini也是这样，不知道有没有例外
                        throw makeError<UnimplError>("Read-only variable cannot be modified.");
                    }
                    case AutoDecreaseFirst:
                    case AutoIncreaseFirst:
                    case AutoIncreaseLast:
                    case AutoDecreaseLast:
                    default: {
                        //先自增/减,再赋值，似乎没有意义，所以忽略
                        if (zenvarNode->bAttr) {
                            //属性的赋值不能修改其原来的维度。
                            //assert(res.value.size() <= var.value.size());
                            std::visit([&](auto& res_value, auto& var_value) {
                                using T1 = std::decay_t<decltype(res_value)>;
                                using E1 = typename T1::value_type;
                                using T2 = std::decay_t<decltype(var_value)>;
                                using E2 = typename T2::value_type;

                                constexpr bool are_int_or_float_v =
                                    (std::is_same_v<E1, int> || std::is_same_v<E1, float>) &&
                                    (std::is_same_v<E2, int> || std::is_same_v<E2, float>);

                                if constexpr (std::is_same_v<E1, E2> || are_int_or_float_v) {
                                    if (res_value.size() < var_value.size()) {
                                        //如果右边的值的容器大小比当前赋值属性要小，很可能是单值，先只考虑这种情况。
                                        assert(res_value.size() == 1);
                                        std::fill(var_value.begin(), var_value.end(), res_value[0]);
                                    }
                                    else {
                                        var = std::move(res);
                                    }
                                }
                                else {
                                    throw makeError<UnimplError>("cannot assign when the type is not convertiable");
                                }
                                }, res.value, var.value);
                            return ZfxVariable();
                        }
                        else {
                            var = std::move(res);
                            return ZfxVariable();
                        }
                    }
                    }
                }
                break;
            }
            case DECLARE: {
                //变量定义
                int nChildren = root->children.size();
                if (nChildren != 2 && nChildren != 3) {
                    throw makeError<UnimplError>("args of DECLARE");
                }
                std::shared_ptr<ZfxASTNode> typeNode = root->children[0];
                std::shared_ptr<ZfxASTNode> nameNode = root->children[1];
                ZfxVariable newvar;
                bool bOnlyDeclare = nChildren == 2;
                operatorVals vartype = typeNode->opVal;
                if (bOnlyDeclare) {
                    switch (vartype)
                    {
                    case TYPE_INT: {
                        newvar.value = std::vector<int>{ 0 };
                        break;
                    }
                    case TYPE_INT_ARR: {
                        newvar.value = std::vector<zfxintarr>{ zfxintarr() };
                        break;
                    }
                    case TYPE_FLOAT: {
                        newvar.value = std::vector<float>{ 0.f };
                        break;
                    }
                    case TYPE_FLOAT_ARR: {
                        newvar.value = std::vector<zfxfloatarr>{ zfxfloatarr() };
                        break;
                    }
                    case TYPE_STRING: {
                        newvar.value = std::vector<std::string>{ "" };
                        break;
                    }
                    case TYPE_STRING_ARR: {
                        newvar.value = std::vector<zfxstringarr>{ zfxstringarr() };
                        break;
                    }
                    case TYPE_VECTOR2: {
                        newvar.value = std::vector<glm::vec2>{ glm::vec2() };
                        break;
                    }
                    case TYPE_VECTOR2_ARR: {
                        newvar.value = std::vector<zfxvec2arr>{ zfxvec2arr() };
                        break;
                    }
                    case TYPE_VECTOR3: {
                        newvar.value = std::vector<glm::vec3>{ glm::vec3() };
                        break;
                    }
                    case TYPE_VECTOR3_ARR: {
                        newvar.value = std::vector<zfxvec3arr>{ zfxvec3arr() };
                        break;
                    }
                    case TYPE_VECTOR4:  newvar.value = std::vector<glm::vec4>{ glm::vec4() }; break;
                    case TYPE_VECTOR4_ARR: {
                        newvar.value = std::vector<zfxvec4arr>{ zfxvec4arr() };
                        break;
                    }
                    case TYPE_MATRIX2:  newvar.value = std::vector<glm::mat2>{ glm::mat2() }; break;
                    case TYPE_MATRIX3:  newvar.value = std::vector<glm::mat3>{ glm::mat3() }; break;
                    case TYPE_MATRIX4:  newvar.value = std::vector<glm::mat4>{ glm::mat4() }; break;
                    }
                }
                else {
                    std::shared_ptr<ZfxASTNode> valueNode = root->children[2];
                    newvar = execute(valueNode, filter, pContext);
                }

                std::string varname = get_zfxvar<std::string>(nameNode->value);
                //暂时不考虑自定义结构体
                bool bret = declareVariable(varname);
                if (!bret) {
                    throw makeError<UnimplError>("assign variable failed.");
                }

                validateVar(vartype, newvar);

                bret = assignVariable(varname, newvar, pContext);
                if (!bret) {
                    throw makeError<UnimplError>("assign variable failed.");
                }
                break;
            }
            case FUNC: {
                //函数
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                const std::string& funcname = get_zfxvar<std::string>(root->value);
                if (funcname == "dot") {
                    return calc_exp(args[0], args[1], filter, glmdot());
                }
                ZfxVariable result = eval(funcname, args, filter, pContext);
                return result;
            }
            case UNARY_EXP: {
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                if (args.size() != 1) {
                    throw makeError<UnimplError>("num args of unary op should be 1");
                }
                if (root->opVal != NOT) {
                    throw makeError<UnimplError>("unknown unary op.");
                }

                const ZfxVariable& arg = args[0];
                ZfxVariable result;

                std::visit([&](const auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    const int N = vec.size();

                    if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                        auto _vec = std::vector<int>(N);
                        for (int i = 0; i < N; i++) {
                            _vec[i] = vec[i];
                        }
                        result.value = _vec;
                    }
                    else if constexpr (std::is_same_v<E, std::string>) {
                        auto _vec = std::vector<int>(N);
                        for (int i = 0; i < N; i++) {
                            _vec[i] = !vec[i].empty();
                        }
                        result.value = _vec;
                    }
                    else {
                        //TODO: vectype
                        throw makeError<UnimplError>("not support of nor operator for other types.");
                    }
                }, arg.value);

                return result;
            }
            case FOUROPERATIONS: {
                //四则运算+ - * / %
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                if (args.size() != 2) {
                    throw makeError<UnimplError>("op args");
                }
                switch (root->opVal) {
                case PLUS:  return calc_exp(args[0], args[1], filter, std::plus());
                case MINUS: return calc_exp(args[0], args[1], filter, std::minus());
                case MUL:   return calc_exp(args[0], args[1], filter, std::multiplies());
                case DIV:   return calc_exp(args[0], args[1], filter, std::divides());
                case MOD:   return calc_exp(args[0], args[1], filter, std::modulus());
                case AND:   return calc_exp(args[0], args[1], filter, std::logical_and());
                case OR:    return calc_exp(args[0], args[1], filter, std::logical_or());
                default:
                    throw makeError<UnimplError>("op error");
                }
            }
            case NEGATIVE: {
                if (root->children.size() != 1) {
                    throw makeError<UnimplError>("NEGATIVE number is missing");
                }

                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                const ZfxVariable& arg = args[0];

                ZfxVariable result;
                std::visit([&](auto& vec) {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;

                    const int N = vec.size();

                    if constexpr (std::is_same_v<E, int> || std::is_same_v<E, float>) {
                        std::vector<E> newvec(N);
                        for (int i = 0; i < N; i++)
                            newvec[i] = -1 * vec[i];
                        result.value = std::move(newvec);
                    }
                    else if constexpr (std::is_same_v<E, glm::vec2>) {
                        std::vector<E> newvec(N);
                        for (int i = 0; i < N; i++) {
                            newvec[i] = glm::vec2(-1*vec[i][0], -1*vec[i][1]);
                        }
                        result.value = std::move(newvec);
                    }
                    else if constexpr (std::is_same_v<E, glm::vec3>) {
                        std::vector<E> newvec(N);
                        for (int i = 0; i < N; i++) {
                            newvec[i] = glm::vec3(-1 * vec[i][0], -1 * vec[i][1], -1 * vec[i][2]);
                        }
                        result.value = std::move(newvec);
                    }
                    else if constexpr (std::is_same_v<E, glm::vec4>) {
                        std::vector<E> newvec(N);
                        for (int i = 0; i < N; i++) {
                            newvec[i] = glm::vec4(-1 * vec[i][0], -1 * vec[i][1], -1 * vec[i][2], -1 * vec[i][3]);
                        }
                        result.value = std::move(newvec);
                    }
                    else {
                        throw makeError<UnimplError>("not support type in `NEGATIVE`");
                    }
                    }, arg.value);
                return result;
            }
            case ATTR_VISIT: {
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("op args at attr visit");
                }
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);

                return std::visit([&](auto& vec, const auto& arg_visit_attr) -> ZfxVariable {
                    using T = std::decay_t<decltype(vec)>;
                    using E = typename T::value_type;
                    using T2 = std::decay_t<decltype(arg_visit_attr)>;
                    using E2 = typename T2::value_type;

                    if constexpr (std::is_same_v<E2, std::string>) {
                        const std::string& visit_attr = arg_visit_attr[0];

                        if constexpr (std::is_same_v<E, ZfxLValue>) {
                            return std::visit([&](auto&& nodeparam) -> ZfxVariable {
                                using E3 = std::decay_t<decltype(nodeparam)>;
                                if constexpr (std::is_same_v<E3, ParamPrimitive>) {
                                    if (visit_attr == "value") {
                                        return anyToZfxVector(nodeparam.defl);
                                    }
                                    else if (visit_attr == "connected") {
                                        return std::vector<int>{ !nodeparam.links.empty() };
                                    }
                                    else if (visit_attr == "x") {
                                        //TODO
                                        return std::vector<float>{ 0.f };
                                    }
                                    else if (visit_attr == "y") {
                                        return std::vector<float>{0.f};
                                    }
                                    else if (visit_attr == "z") {
                                        return std::vector<float>{0.f};
                                    }
                                    else if (visit_attr == "w") {
                                        return std::vector<float>{0.f};
                                    }
                                    else {
                                        //unknown attr
                                        throw makeError<UnimplError>("unknown attr when visit nodeparam");
                                    }
                                }
                                else if constexpr (std::is_same_v<E3, ParamObject>) {
                                    if (visit_attr == "connected") {
                                        return std::vector<int>{ !nodeparam.links.empty() };
                                    }
                                    else {
                                        throw makeError<UnimplError>("unknown attr when visit nodeparam");
                                    }
                                }
                                }, vec[0].var);
                        }
                        else if constexpr (std::is_same_v<E, glm::vec2> ||
                            std::is_same_v<E, glm::vec3> ||
                            std::is_same_v<E, glm::vec4>)
                        {
                            ZfxVariable ret;
                            int nVarSize = vec.size();
                            std::vector<float> ret_vec(nVarSize);
                            int comp = 0;
                            if (visit_attr == "x") comp = 0;
                            else if (visit_attr == "y") comp = 1;
                            else if (visit_attr == "z") comp = 2;

                            for (int i = 0; i < nVarSize; i++) {
                                ret_vec[i] = vec[i][comp];
                            }
                            ret.value = std::move(ret_vec);
                            return ret;
                        }
                        else {
                            throw makeError<UnimplError>("not support type in scope `ATTR_VISIT`");
                        }
                    }
                    else {
                        throw makeError<UnimplError>("visit attr should be `string` type");
                    }
                    }, args[0].value, args[1].value);
            }
            case COMPOP: {
                //操作符
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                if (args.size() != 2) {
                    throw makeError<UnimplError>("compare op args");
                }
                switch (root->opVal) {
                case Less:          return calc_exp(args[0], args[1], filter, std::less());
                case LessEqual:     return calc_exp(args[0], args[1], filter, std::less_equal());
                case Greater:       return calc_exp(args[0], args[1], filter, std::greater());
                case GreaterEqual:  return calc_exp(args[0], args[1], filter, std::greater_equal());
                case Equal:         return calc_exp(args[0], args[1], filter, std::equal_to());
                case NotEqual:      return calc_exp(args[0], args[1], filter, std::not_equal_to());
                default:
                    throw makeError<UnimplError>("compare op error");
                }
            }
            case ARRAY:{
                return parseArray(root, filter, pContext);
            }
            case PLACEHOLDER:{
                return ZfxVariable();
            }
            case CONDEXP: {
                //条件表达式
                if (root->children.size() != 3) {
                    throw makeError<UnimplError>("cond exp args");
                }
                auto pCondExp = root->children[0];
                const ZfxVariable& cond = execute(pCondExp, filter, pContext);
                if (cond.size() == 1) {
                    //单值，不是向量
                    ZfxElemFilter newFilter, elseFilter;
                    if (hasTrue(cond, filter, newFilter, elseFilter)) {
                        auto pCodesExp = root->children[1];
                        return execute(pCodesExp, newFilter, pContext);
                    }
                    else {
                        auto pelseExp = root->children[2];
                        return execute(pelseExp, filter, pContext);
                    }
                }
                else {
                    //向量的情况，每个分支都要执行，然后合并
                    ZfxElemFilter ifFilter, elseFilter;
                    ZfxVariable switch1, switch2, ret;
                    hasTrue(cond, filter, ifFilter, elseFilter);
  
                    auto pifExp = root->children[1];
                    switch1 = execute(pifExp, ifFilter, pContext);

                    auto pelseExp = root->children[2];
                    switch2 = execute(pelseExp, elseFilter, pContext);

                    int n = cond.size();

                    std::visit([&](const auto& switch1_value, const auto& switch2_value) {
                        using T1 = std::decay_t<decltype(switch1_value)>;
                        using E1 = typename T1::value_type;
                        using T2 = std::decay_t<decltype(switch2_value)>;
                        using E2 = typename T2::value_type;

                        //先拿E1
                        if constexpr (std::is_same_v<E1, E2>) {
                            std::vector<E1> result(n);
                            for (int i = 0; i < n; i++) {
                                result[i] = ifFilter[i] ? switch1_value[i] : switch2_value[i];
                            }
                            ret.value = std::move(result);
                        }
                        else {
                            throw makeError<UnimplError>("different type in `CONDEXP`");
                        }
                        }, switch1.value, switch2.value);

                    return ret;
                }
                throw makeError<UnimplError>("error condition on condexp");
            }
            case IF:{
                if (root->children.size() != 2 && root->children.size() != 3) {
                    throw makeError<UnimplError>("if cond failed.");
                }
                auto pCondExp = root->children[0];
                //todo: self inc
                const ZfxVariable& cond = execute(pCondExp, filter, pContext);
                if (cond.size() == 1) {//不是向量的情况
                    ZfxElemFilter newFilter, elseFilter;
                    if (hasTrue(cond, filter, newFilter, elseFilter)) {
                        auto pCodesExp = root->children[1];
                        execute(pCodesExp, newFilter, pContext);
                    } else if (root->children.size() == 3) {
                        auto pelseExp = root->children[2];
                        execute(pelseExp, filter, pContext);
                    }
                } else {//向量的情况，每个分支都要执行
                    ZfxElemFilter ifFilter, elseFilter;
                    if (hasTrue(cond, filter, ifFilter, elseFilter)) {
                        auto pCodesExp = root->children[1];
                        execute(pCodesExp, ifFilter, pContext);
                    }
                    if (root->children.size() == 3) {
                        auto pelseExp = root->children[2];
                        execute(pelseExp, elseFilter, pContext);
                    }
                }

                break;
            }
            case FOR:{
                if (root->children.size() != 4) {
                    throw makeError<UnimplError>("for failed.");
                }
                auto forBegin = root->children[0];
                auto forCond = root->children[1];
                auto forStep = root->children[2];
                auto loopContent = root->children[3];

                //压栈
                pushStack();
                scope_exit sp([this]() {this->popStack(); });

                //查看begin语句里是否有定义语句，赋值语句
                switch (forBegin->type)
                {
                case DECLARE:
                {
                    execute(forBegin, filter, pContext);
                    break;
                }
                case ASSIGNMENT:
                {
                    execute(forBegin, filter, pContext);
                    break;
                }
                case PLACEHOLDER:
                default:
                    //填其他东西没有意义，甚至可能导致解析出错。
                    execute(forBegin, filter, pContext);
                    break;
                }

                ZfxVariable cond = execute(forCond, filter, pContext);
                ZfxElemFilter ifFilter, elseFilter;
                while (hasTrue(cond, filter, ifFilter, elseFilter)) {
                    //TODO: check the passed element and mark in the newFilter.
                    execute(loopContent, ifFilter, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return ZfxVariable();

                    execute(forStep, ifFilter, pContext);
                    cond = execute(forCond, ifFilter, pContext);
                }
                break;
            }
            case FOREACH:{
                //对应文法：FOREACH LPAREN foreach-step COLON zenvar RPAREN code-block
                //foreach-step 可能有一个或两个子成员，并返回vec<ASTNode>
                int nChild = root->children.size();
                if (nChild == 3 || nChild == 4) {
                    //没有索引
                    auto idxNode = nChild == 3 ? nullptr : root->children[0];
                    auto varNode = nChild == 3 ? root->children[0] : root->children[1];
                    auto arrNode = nChild == 3 ? root->children[1] : root->children[2];
                    auto codeSeg = nChild == 3 ? root->children[2] : root->children[3];
                    if (!varNode || !arrNode || !codeSeg)
                        throw makeError<UnimplError>("elements on foreach error.");

                    const ZfxVariable& arr = execute(arrNode, filter, pContext);

                    //压栈
                    pushStack();
                    scope_exit sp([this]() {this->popStack(); });

                    //定义idxNode所指向的名称为值 i
                    //定义varNode所指向的名称为值 arrtest[i]
                    std::string idxName;
                    if (idxNode) {
                        idxName = get_zfxvar<std::string>(idxNode->value);
                        declareVariable(idxName);
                    }
                    const std::string& varName = get_zfxvar<std::string>(varNode->value);
                    declareVariable(varName);

                    std::visit([&](auto& vec) {
                        using T = std::decay_t<decltype(vec)>;
                        using E = typename T::value_type;
                        if constexpr (std::is_same_v<E, zfxintarr> ||
                            std::is_same_v<E, zfxfloatarr> ||
                            std::is_same_v<E, zfxstringarr>)
                        {
                            using E2 = typename E::value_type;
                            for (const auto& val : vec) {
                                for (int i = 0; i < val.size(); i++) {
                                    //修改变量和索引的值为i, arrtest[i];
                                    if (idxNode) {
                                        ZfxVariable zfxvar;
                                        zfxvar.value = std::vector<int>{ i };
                                        assignVariable(idxName, zfxvar, pContext);
                                    }

                                    ZfxVariable zfxvar;
                                    zfxvar.value = std::vector<E2>{ val[i] };
                                    assignVariable(varName, zfxvar, pContext);

                                    //修改定义后，再次运行code
                                    execute(codeSeg, filter, pContext);

                                    //检查是否有跳转, continue在execute内部已经跳出了，这里不需要处理
                                    if (pContext->jumpFlag == JUMP_BREAK ||
                                        pContext->jumpFlag == JUMP_RETURN) {
                                        return;
                                    }
                                }
                            }
                        }
                        else if constexpr (std::is_same_v<E, glm::vec2> ||
                            std::is_same_v<E, glm::vec3> ||
                            std::is_same_v<E, glm::vec4>)
                        {
                            using E2 = typename E::value_type;
                            for (const auto& val : vec) {
                                for (int i = 0; i < val.length(); i++) {
                                    //修改变量和索引的值为i, arrtest[i];
                                    if (idxNode) {
                                        ZfxVariable zfxvar;
                                        zfxvar.value = std::vector<int>{ i };
                                        assignVariable(idxName, zfxvar, pContext);
                                    }

                                    ZfxVariable zfxvar;
                                    zfxvar.value = std::vector<E2>{ val[i] };
                                    assignVariable(varName, zfxvar, pContext);

                                    //修改定义后，再次运行code
                                    execute(codeSeg, filter, pContext);

                                    //检查是否有跳转, continue在execute内部已经跳出了，这里不需要处理
                                    if (pContext->jumpFlag == JUMP_BREAK ||
                                        pContext->jumpFlag == JUMP_RETURN) {
                                        return;
                                    }
                                }
                            }
                        }
                        else {
                            throw makeError<UnimplError>("not support type in `FOREACH`");
                        }
                        }, arr.value);

                    return ZfxVariable();
                }
                else {
                    throw makeError<UnimplError>("foreach error.");
                }
                break;
            }
            case WHILE:{
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("while failed.");
                }
   
                auto forCond = root->children[0];
                auto loopContent = root->children[1];

                //压栈
                pushStack();
                scope_exit sp([this]() {this->popStack(); });

                auto cond = execute(forCond, filter, pContext);
                ZfxElemFilter newFilter, elseFilter;
                while (hasTrue(cond, filter, newFilter, elseFilter)) {
                    execute(loopContent, newFilter, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return ZfxVariable();

                    cond = execute(forCond, newFilter, pContext);
                }
                break;
            }
            case DOWHILE:{
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("while failed.");
                }

                auto forCond = root->children[1];
                auto loopContent = root->children[0];

                //??
                pushStack();
                scope_exit sp([this]() {this->popStack(); });
                ZfxVariable cond;

                ZfxElemFilter newFilter = filter;
                ZfxElemFilter newFilter2, elsefilter;

                do {
                    newFilter = newFilter2;
                    execute(loopContent, newFilter, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return ZfxVariable();
                    cond = execute(forCond, newFilter, pContext);
                } while (hasTrue(cond, newFilter, newFilter2, elsefilter));

                break;
            }
            case CODEBLOCK:{
                pushStack();
                scope_exit sp([this]() {this->popStack(); });
                for (auto pSegment : root->children) {
                    const ZfxVariable& res = execute(pSegment, filter, pContext);
                    if (pContext->bSingleFmla)
                        return res;

                    if (pContext->jumpFlag == JUMP_BREAK ||
                        pContext->jumpFlag == JUMP_CONTINUE ||
                        pContext->jumpFlag == JUMP_RETURN) {
                        return ZfxVariable();
                    }
                }
                break;
            }
            case JUMP:{
                pContext->jumpFlag = root->opVal;
                break;
            }
            case VARIABLETYPE:{
                //变量类型，比如int vector3 float string等
            }
            default: {
                break;
            }
        }
        return ZfxVariable();
    }

    std::set<std::pair<std::string, std::string>>
        FunctionManager::getReferSources(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext)
    {
        if (!root) {
            return {};
        }

        std::set<std::pair<std::string, std::string>> paths;
        if (nodeType::FUNC != root->type)
        {
            for (auto _childNode : root->children)
            {
                std::set<std::pair<std::string, std::string>> _paths = getReferSources(_childNode, pContext);
                if (!_paths.empty()) {
                    paths.insert(_paths.begin(), _paths.end());
                }
            }
        }
        else
        {
            const std::string& funcname = std::get<std::string>(root->value);
            if (funcname == "ref") {
                if (root->children.size() != 1) {
                    //可能只是编辑时候无意输入，没必要抛异常
                    return {};
                }
                const zeno::zfxvariant& res = calc(root->children[0], pContext);
                const std::string ref = std::holds_alternative<std::string>(res) ? std::get<std::string>(res) : "";
                //收集ref信息源，包括源节点和参数
                std::string paramname, _;
                auto spNode = zfx::getNodeAndParamFromRefString(ref, pContext, paramname, _);
                if (spNode)
                    paths.insert(std::make_pair(spNode->get_uuid_path(), paramname));
            }
            else {
                //函数参数也可能调用引用：
                for (auto paramNode : root->children)
                {
                    std::set<std::pair<std::string, std::string>> _paths;
                    if (paramNode->type ==  nodeType::STRING) {
                        const zeno::zfxvariant& res = calc(paramNode, pContext);
                        const std::string ref = std::holds_alternative<std::string>(res) ? std::get<std::string>(res) : "";
                        if (std::regex_search(ref, refPattern)) {
                            std::string paramname, _;
                            auto spNode = zfx::getNodeAndParamFromRefString(ref, pContext, paramname, _);
                            if (spNode)
                                paths.insert(std::make_pair(spNode->get_uuid_path(), paramname));
                        }
                    } else {
                        _paths = getReferSources(paramNode, pContext);
                    }
                    if (!_paths.empty()) {
                        paths.insert(_paths.begin(), _paths.end());
                    }
                }
            }
        }
        return paths;
    }

    zfxvariant FunctionManager::calc(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext) {
        if (root)
        {
            switch (root->type)
            {
            case nodeType::NUMBER:
            case nodeType::STRING: return root->value;
            case nodeType::ZENVAR:
            {
                const std::string& var = std::get<std::string>(root->value);
                if (var == "F") {
                    return (float)zeno::getSession().globalState->getFrameId();
                }
                else if (var == "FPS") {
                    //TODO
                    return zfxvariant();
                }
                else if (var == "T") {
                    //TODO
                    return zfxvariant();
                }
            }
            case nodeType::FOUROPERATIONS:
            {
                if (root->children.size() != 2)
                {
                    throw makeError<UnimplError>();
                }
                zfxvariant lhs = calc(root->children[0], pContext);
                zfxvariant rhs = calc(root->children[1], pContext);

                const std::string& var = std::get<std::string>(root->value);
                if (var == "+") {
                    //TODO: vector
                    return std::get<float>(lhs) + std::get<float>(rhs);
                }
                else if (var == "-") {
                    return std::get<float>(lhs) - std::get<float>(rhs);
                }
                else if (var == "*") {
                    return std::get<float>(lhs) * std::get<float>(rhs);
                }
                else if (var == "/") {
                    if (std::get<float>(rhs) == 0)
                        throw makeError<UnimplError>();
                    return std::get<float>(lhs) / std::get<float>(rhs);
                }
                else {
                    return zfxvariant();
                }
            }
            case nodeType::NEGATIVE:
            {
                if (root->children.size() != 1)
                {
                    throw makeError<UnimplError>("NEGATIVE number is missing");
                }
                zfxvariant val = calc(root->children[0], pContext);
                val = std::visit([&](auto& arg) -> zfxvariant {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                        return -1 * arg;
                    }
                    else if constexpr (std::is_same_v<T, glm::vec2>)
                    {
                        return glm::vec2{ -1 * arg[0], -1 * arg[1] };
                    }
                    else if constexpr (std::is_same_v<T, glm::vec3>)
                    {
                        return glm::vec3{ -1 * arg[0], -1 * arg[1], -1 * arg[2] };
                    }
                    else if constexpr (std::is_same_v<T, glm::vec4>)
                    {
                        return glm::vec4{ -1 * arg[0], -1 * arg[1], -1 * arg[2], -1 * arg[3] };
                    }
                    else {
                        throw makeError<UnimplError>("NEGATIVE number type is invalid");
                    }
                }, val);
                return val;
            }
            case nodeType::FUNC:
            {
                ZfxElemFilter filter;
                filter.push_back(1);
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                const std::string& funcname = get_zfxvar<std::string>(root->value);
                ZfxVariable result = eval(funcname, args, filter, pContext);
                return std::visit([&](auto& vec)->zfxvariant {
                    return vec[0];
                    }, result.value);
            }
            }
        }
        return zfxvariant();
    }



    ZfxVariable FunctionManager::eval(const std::string& funcname, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
        return callFunction(funcname, args, filter, pContext);
    }

}