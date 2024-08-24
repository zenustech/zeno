#include <zeno/core/FunctionManager.h>
#include <zeno/core/ReferManager.h>
#include <zeno/core/Session.h>
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
#include <zeno/utils/vectorutil.h>


namespace zeno {

    FunctionManager::FunctionManager() {
        init();
    }

    std::vector<std::string> FunctionManager::getCandidates(const std::string& prefix, bool bFunc) const {
        std::vector<std::string> candidates;
        if (bFunc && prefix.empty())
            return candidates;

        if (bFunc) {
            for (auto& [k, v] : m_funcs) {
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
        auto iter = m_funcs.find(funcName);
        if (iter == m_funcs.end()) {
            bExist = false;
            return "";
        }
        bExist = true;
        return iter->second.tip;
    }

    ZENO_API FUNC_INFO FunctionManager::getFuncInfo(const std::string& funcName) const {
        auto iter = m_funcs.find(funcName);
        if (iter == m_funcs.end()) {
            return FUNC_INFO();
        }
        return iter->second;
    }

    float FunctionManager::callRef(const std::string& ref, ZfxContext* pContext) {
        //TODO: vec type.
        //TODO: resolve with zeno::reflect::any
        return 0;
#if 0
        std::string fullPath, graphAbsPath;

        if (ref.empty()) {
            throw makeError<UnimplError>();
        }

        auto thisNode = pContext->spNode.lock();
        const std::string& thisnodePath = thisNode->get_path();
        graphAbsPath = thisnodePath.substr(0, thisnodePath.find_last_of('/'));

        if (ref.front() == '/') {
            fullPath = ref;
        } else {
            fullPath = graphAbsPath + "/" + ref;
        }

        int idx = fullPath.find_last_of('/');
        if (idx == std::string::npos) {
            throw makeError<UnimplError>();
        }

        const std::string& nodePath = fullPath.substr(idx + 1);

        idx = nodePath.find('.');
        if (idx == std::string::npos) {
            throw makeError<UnimplError>();
        }
        std::string nodename = nodePath.substr(0, idx);
        std::string parampath = nodePath.substr(idx + 1);

        std::string nodeAbsPath = graphAbsPath + '/' + nodename;
        std::shared_ptr<INode> spNode = zeno::getSession().mainGraph->getNodeByPath(nodeAbsPath);

        if (!spNode) {
            throw makeError<UnimplError>();
        }

        auto items = split_str(parampath, '.');
        std::string paramname = items[0];

        bool bExist = false;
        ParamPrimitive paramData = spNode->get_input_prim_param(paramname, &bExist);
        if (!bExist)
            throw makeError<UnimplError>();

        if (items.size() == 1) {
            if (std::holds_alternative<int>(paramData.defl)) {
                return std::get<int>(paramData.defl);
            }
            else if (std::holds_alternative<float>(paramData.defl)) {
                return std::get<float>(paramData.defl);
            }
            else {
                throw makeError<UnimplError>();
            }
        }

        if (items.size() == 2 &&
            (paramData.type == Param_Vec2f || paramData.type == Param_Vec2i ||
                paramData.type == Param_Vec3f || paramData.type == Param_Vec3i ||
                paramData.type == Param_Vec4f || paramData.type == Param_Vec4i))
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
            if (paramData.type == Param_Vec2f || paramData.type == Param_Vec2i) {
                if (idx < 2) {
                    return paramData.type == Param_Vec2f ? std::get<vec2f>(paramData.defl)[idx] :
                        std::get<vec2i>(paramData.defl)[idx];
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
            if (paramData.type == Param_Vec3f || paramData.type == Param_Vec3i) {
                if (idx < 3) {
                    return paramData.type == Param_Vec3f ? std::get<vec3f>(paramData.defl)[idx] :
                        std::get<vec3i>(paramData.defl)[idx];
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
            if (paramData.type == Param_Vec4f || paramData.type == Param_Vec4i) {
                if (idx < 4) {
                    return paramData.type == Param_Vec4f ? std::get<vec4f>(paramData.defl)[idx] :
                        std::get<vec4i>(paramData.defl)[idx];
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
        }
        throw makeError<UnimplError>();
#endif
    }

    static int getElementCount(std::shared_ptr<IObject> spObject, ZfxRunOver runover) {
        switch (runover)
        {
        case RunOver_Points: {
            if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(spObject)) {
                return spGeo->get_point_count();
            }
            else if (auto spPrim = std::dynamic_pointer_cast<PrimitiveObject>(spObject)) {
                return spPrim->verts->size();
            }
            else {
                return 0;
            }
        }
        case RunOver_Face:
            if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(spObject)) {
                return spGeo->get_face_count();
            }
            else if (auto spPrim = std::dynamic_pointer_cast<PrimitiveObject>(spObject)) {
                if (spPrim->tris.size() > 0)
                    return spPrim->tris.size();
                else
                    return spPrim->polys.size();
            }
            else {
                return 0;
            }
        case RunOver_Geom: {
            //only one element
            return 1;
        }
        }
    }

    void FunctionManager::executeZfx(std::shared_ptr<ZfxASTNode> root, ZfxContext* pCtx) {
        //printSyntaxTree(root, pCtx->code);
        assert(pCtx->spObject);
        if (pCtx->spObject) {
            int nFilterSize = getElementCount(pCtx->spObject, pCtx->runover);
            ZfxElemFilter filter(nFilterSize, 1);
            scope_exit sp([&] {m_globalAttrCached.clear(); });
            execute(root, filter, pCtx);
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

    template<typename Operator>
    ZfxVariable calc_exp(const ZfxVariable& lhs, const ZfxVariable& rhs, const ZfxElemFilter& filter, Operator method) {

        int N1 = lhs.value.size();
        int N2 = rhs.value.size();
        int minsize = min(N1, N2);
        int maxsize = max(N1, N2);
        if (N1 != N2) {
            if (minsize != 1)
                throw makeError<UnimplError>("size invalidation on calc_exp");
        }

        ZfxVariable res;
        res.value.resize(maxsize);

        for (int i = 0; i < maxsize; i++)
        {
            if (!filter[i])
                continue;

            const zfxvariant& _lhs = N1 <= i ? lhs.value[0] : lhs.value[i];
            const zfxvariant& _rhs = N2 <= i ? rhs.value[0] : rhs.value[i];

            res.value[i] = std::visit([method](auto&& lval, auto&& rval)->zfxvariant {
                using T = std::decay_t<decltype(lval)>;
                using E = std::decay_t<decltype(rval)>;
                using Op = std::decay_t<decltype(method)>;

                if constexpr (std::is_same_v<Op, std::modulus<>>) {
                    if constexpr (std::is_same_v<T, int> && std::is_same_v<E, int>) {
                        return method((int)lval, (int)rval);
                    }
                    else if constexpr (std::is_same_v<T, int> && std::is_same_v<E, float>) {
                        return method(lval, (int)rval);
                    }
                    else if constexpr (std::is_same_v<T, float> && std::is_same_v<E, int>) {
                        return method((int)lval, rval);
                    }
                    else if constexpr (std::is_same_v<T, float> && std::is_same_v<E, float>) {
                        return method((int)lval, (int)rval);
                    }
                    throw UnimplError("");
                }
                else if constexpr (std::is_same_v<T, int>) {
                    if constexpr (std::is_same_v<E, int>) {
                        return method(lval, rval);
                    }
                    else if constexpr (std::is_same_v<E, float>) {
                        return method(E(lval), rval);
                    }
                    else {
                        //暂不考虑一个元素和一个矩阵相加的情况。
                        throw UnimplError("");
                    }
                }
                else if constexpr (std::is_same_v<T, float>) {
                    if constexpr (std::is_same_v<E, int>) {
                        return method(lval, T(rval));
                    }
                    else if constexpr (std::is_same_v<E, float>) {
                        return method(lval, rval);
                    }
                    throw UnimplError("");
                }
                else if constexpr (std::is_same_v<T, glm::vec2> && std::is_same_v<T, E> ||
                    std::is_same_v<T, glm::vec3> && std::is_same_v<T, E> ||
                    std::is_same_v<T, glm::vec4> && std::is_same_v<T, E>)
                {
                    if constexpr (std::is_same_v<Op, std::less_equal<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::less<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::greater<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::greater_equal<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::logical_or<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::logical_and<>>) {
                        throw UnimplError("");
                    }
                    else
                    {
                        return method(lval, rval);
                    }
                }
                else if constexpr (std::is_same_v<T, glm::mat3> && std::is_same_v<T, E> ||
                    std::is_same_v<T, glm::mat4> && std::is_same_v<T, E> ||
                    std::is_same_v<T, glm::mat2> && std::is_same_v<T, E>) {

                    if constexpr (std::is_same_v<Op, std::less_equal<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::less<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::greater<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::greater_equal<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::multiplies<>>)
                    {
                        //glm的实现里，乘法是顺序相反的，比如A*B, 其实是我们理解的B * A.
                        return method(rval, lval);
                    }
                    else if constexpr (std::is_same_v<Op, std::logical_or<>>) {
                        throw UnimplError("");
                    }
                    else if constexpr (std::is_same_v<Op, std::logical_and<>>) {
                        throw UnimplError("");
                    }
                    else {
                        return method(lval, rval);
                    }
                }
                else {
                    throw UnimplError("");
                }
            }, _lhs, _rhs);
        }
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
        for (int i = 0; i < zfxarr.value.size(); i++) {
            auto& arr = zfxarr.value[i];
            auto& val = zfxvalue.value.size() == 1 ? zfxvalue.value[0] : zfxvalue.value[i];
            int idx = idxarr.value.size() == 1 ? get_zfxvar<int>(idxarr.value[0]) : get_zfxvar<int>(idxarr.value[i]);

            std::visit([idx](auto& arr, auto& value) {
                using T = std::decay_t<decltype(arr)>;
                using V = std::decay_t<decltype(value)>;
                //mat取一层索引可能就是vec...
                if constexpr ((std::is_same_v<T, zfxintarr> || std::is_same_v<T, zfxfloatarr>) &&
                    std::is_arithmetic_v<V>) {
                    arr[idx] = value;
                }
                else if constexpr ((std::is_same_v<T, glm::vec2> ||
                    std::is_same_v<T, glm::vec3> ||
                    std::is_same_v<T, glm::vec4>) && std::is_same_v<V, float>) {
                    arr[idx] = value;
                }
            }, arr, val);
        }
    }

    static ZfxVariable get_array_element(const ZfxVariable& arr, const ZfxVariable& varidx) {
        int idx = 0;
        if (varidx.value.size() == 1) {
            idx = get_zfxvar<int>(varidx.value[0]);
        }
        else {
            assert(arr.value.size() == varidx.value.size());
        }

        ZfxVariable res;
        for (int i = 0; i < arr.value.size(); i++) {
            if (varidx.value.size() == 1)
                idx = get_zfxvar<int>(varidx.value[0]);
            else
                idx = get_zfxvar<int>(varidx.value[i]);

            res.value.push_back(std::visit([idx](auto&& arg) -> zfxvariant {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, zfxintarr> ||
                    std::is_same_v<T, zfxfloatarr> ||
                    std::is_same_v<T, zfxstringarr> ||
                    std::is_same_v<T, glm::vec2> ||
                    std::is_same_v<T, glm::vec3> ||
                    std::is_same_v<T, glm::vec4> ||
                    std::is_same_v<T, glm::mat2> ||
                    std::is_same_v<T, glm::mat3> ||
                    std::is_same_v<T, glm::mat4>
                    ) {
                    return (T(arg))[idx];
                }
                else {
                    throw makeError<UnimplError>("get elemvar from arr");
                }
            }, arr.value[i]));
        }
        return res;
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
        varidx.value.push_back(idx);
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
        varidx.value.push_back(idx);
        set_array_element(arr, varidx, value);
    }

    static void selfIncOrDec(ZfxVariable& var, bool bInc) {
        for (auto& val : var.value) {
            std::visit([bInc](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                    bInc ? (T)arg++ : (T)arg--;
                }
                else {
                    throw makeError<UnimplError>("Type Error");
                }
            }, val);
        }
    }

    std::vector<ZfxVariable> FunctionManager::process_args(std::shared_ptr<ZfxASTNode> parent, ZfxElemFilter& filter, ZfxContext* pContext) {
        std::vector<ZfxVariable> args;
        for (auto pChild : parent->children) {
            ZfxVariable argval = execute(pChild, filter, pContext);
            args.push_back(argval);
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
        if (name.at(0) == '@') {
            auto iter = m_globalAttrCached.find(name);
            if (iter == m_globalAttrCached.end()) {
                const auto& res = m_globalAttrCached.insert(std::make_pair(name, ZfxVariable()));
                iter = res.first;
                iter->second = getAttrValue(name, pContext);
            }
            return iter->second;
        }
        else {
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

    void FunctionManager::validateVar(operatorVals vartype, ZfxVariable& newvars) {
        for (auto& newvar : newvars.value) {
            switch (vartype)
            {
            case TYPE_INT: {
                if (std::holds_alternative<float>(newvar)) {
                    newvar = (int)std::get<float>(newvar);
                }
                else if (std::holds_alternative<int>(newvar)) {

                }
                else {
                    throw makeError<UnimplError>("type dismatch TYPE_INT");
                }
                break;
            }
            case TYPE_INT_ARR: {
                if (std::holds_alternative<zfxfloatarr>(newvar)) {
                    zfxfloatarr floatarr;
                    for (auto&& val : std::get<zfxfloatarr>(newvar))
                        floatarr.push_back(val);
                    newvar = floatarr;
                }
                else if (!std::holds_alternative<zfxintarr>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_INT_ARR");
                }
                break;
            }
            case TYPE_FLOAT: {
                if (std::holds_alternative<float>(newvar)) {

                }
                else if (std::holds_alternative<int>(newvar)) {
                    newvar = (float)std::get<int>(newvar);
                }
                else {
                    throw makeError<UnimplError>("type dismatch TYPE_FLOAT");
                }
                break;
            }
            case TYPE_FLOAT_ARR: {
                if (std::holds_alternative<zfxintarr>(newvar)) {
                    zfxintarr intarr;
                    for (auto&& val : std::get<zfxintarr>(newvar))
                        intarr.push_back(val);
                    newvar = intarr;
                }
                else if (!std::holds_alternative<zfxfloatarr>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_FLOAT_ARR");
                }
                break;
            }
            case TYPE_STRING: {
                if (!std::holds_alternative<std::string>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_STRING");
                }
                break;
            }
            case TYPE_STRING_ARR: {
                if (!std::holds_alternative<zfxstringarr>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_STRING_ARR");
                }
                break;
            }
            case TYPE_VECTOR2: {
                if (std::holds_alternative<zfxfloatarr>(newvar)) {
                    zfxfloatarr arr = std::get<zfxfloatarr>(newvar);
                    if (arr.size() != 2) {
                        throw makeError<UnimplError>("num of elements of arr dismatch");
                    }
                    glm::vec2 vec = { arr[0], arr[1] };
                    newvar = vec;
                }
                else if (std::holds_alternative<glm::vec2>(newvar)) {

                }
                else {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR2");
                }
                break;
            }
            case TYPE_VECTOR3: {
                if (std::holds_alternative<zfxfloatarr>(newvar)) {
                    zfxfloatarr arr = std::get<zfxfloatarr>(newvar);
                    if (arr.size() != 3) {
                        throw makeError<UnimplError>("num of elements of arr dismatch");
                    }
                    glm::vec3 vec = { arr[0], arr[1], arr[2] };
                    newvar = vec;
                }
                else if (std::holds_alternative<glm::vec3>(newvar)) {

                }
                else {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR3");
                }
                break;
            }
            case TYPE_VECTOR4: {
                if (std::holds_alternative<zfxfloatarr>(newvar)) {
                    zfxfloatarr arr = std::get<zfxfloatarr>(newvar);
                    if (arr.size() != 4) {
                        throw makeError<UnimplError>("num of elements of arr dismatch");
                    }
                    glm::vec4 vec = { arr[0], arr[1], arr[2], arr[3] };
                    newvar = vec;
                }
                else if (std::holds_alternative<glm::vec4>(newvar)) {

                }
                else {
                    throw makeError<UnimplError>("type dismatch TYPE_VECTOR4");
                }
                break;
            }
            case TYPE_MATRIX2: {
                if (!std::holds_alternative<glm::mat2>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_MATRIX2");
                }
                break;
            }
            case TYPE_MATRIX3: {
                if (!std::holds_alternative<glm::mat3>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_MATRIX3");
                }
                break;
            }
            case TYPE_MATRIX4: {
                if (!std::holds_alternative<glm::mat4>(newvar)) {
                    throw makeError<UnimplError>("type dismatch TYPE_MATRIX4");
                }
                break;
            }
            }
        }
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

        zfxvariant current;
        operatorVals dataType = UNDEFINE_OP;
        for (int idx = 0; idx < args.size(); idx++) {
            auto& arg = args[idx];
            if (std::holds_alternative<int>(arg.value[0])) {
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_FLOAT_ARR;
                    current = zfxfloatarr();
                }
                if (dataType == TYPE_FLOAT_ARR) {
                    auto& arr = std::get<zfxfloatarr>(current);
                    arr.push_back(std::get<int>(arg.value[0]));
                }
                else {
                    throw makeError<UnimplError>("data type inconsistent");
                }
            }
            else if (std::holds_alternative<float>(arg.value[0])) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_FLOAT_ARR) {
                    throw makeError<UnimplError>("data type inconsistent");
                }
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_FLOAT_ARR;
                    current = zfxfloatarr();
                }
                if (dataType == TYPE_FLOAT_ARR) {
                    auto& arr = std::get<zfxfloatarr>(current);
                    arr.push_back(std::get<float>(arg.value[0]));
                }
            }
            else if (std::holds_alternative<std::string>(arg.value[0])) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_STRING_ARR) {
                    throw makeError<UnimplError>("data type inconsistent");
                }
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_STRING_ARR;
                    current = zfxstringarr();
                }
                if (dataType == TYPE_STRING_ARR) {
                    auto& arr = std::get<zfxstringarr>(current);
                    arr.push_back(std::get<std::string>(arg.value[0]));
                }
            }
            //不考虑intarr，因为glm的vector/matrix都是储存float
            else if (std::holds_alternative<zfxfloatarr>(arg.value[0])) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_MATRIX2 && dataType != TYPE_MATRIX3 && dataType != TYPE_MATRIX4) {
                    throw makeError<UnimplError>("data type inconsistent");
                }

                auto& arr = std::get<zfxfloatarr>(arg.value[0]);

                if (dataType == UNDEFINE_OP) {
                    if (arr.size() == 2) {
                        dataType = TYPE_MATRIX2;
                        current = glm::mat2();
                    }
                    else if (arr.size() == 3) {
                        dataType = TYPE_MATRIX3;
                        current = glm::mat3();
                    }
                    else if (arr.size() == 4) {
                        dataType = TYPE_MATRIX4;
                        current = glm::mat4();
                    }
                }

                //{{0, 1}, {2, 3}}
                if (dataType == TYPE_MATRIX2 && arr.size() == 2 && idx < 2) {
                    auto& mat = std::get<glm::mat2>(current);
                    mat[idx][0] = arr[0];
                    mat[idx][1] = arr[1];
                }
                //{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
                else if (dataType == TYPE_MATRIX3 && arr.size() == 3 && idx < 3) {
                    auto& mat = std::get<glm::mat3>(current);
                    mat[idx][0] = arr[0];
                    mat[idx][1] = arr[1];
                    mat[idx][2] = arr[2];
                }
                //{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}}
                else if (dataType == TYPE_MATRIX4 && arr.size() == 4 && idx < 4) {
                    auto& mat = std::get<glm::mat4>(current);
                    mat[idx][0] = arr[0];
                    mat[idx][1] = arr[1];
                    mat[idx][2] = arr[2];
                    mat[idx][3] = arr[3];
                }
                else {
                    throw makeError<UnimplError>("mat element dims inconsistent");
                }
            }
            else {
                throw makeError<UnimplError>("data type inconsistent");
            }
        }

        ZfxVariable res;
        res.value.push_back(current);
        return res;
    }

    bool FunctionManager::hasTrue(const ZfxVariable& cond, const ZfxElemFilter& filter, ZfxElemFilter& newFilter) const {
        int N = cond.value.size();
        assert(N == filter.size() || N == 1);
        newFilter = filter;
        bool bret = false;
        for (int i = 0; i < cond.value.size(); i++) {
            if (filter[i]) {
                if (get_zfxvar<int>(cond.value[i]) ||
                    get_zfxvar<float>(cond.value[i]))
                {
                    bret = true;
                }
                else
                {
                    newFilter[i] = 0;
                }
            }
        }
        return bret;
    }

    static void commitToObject(std::shared_ptr<IObject> spObject, const ZfxVariable& val, const std::string& attr_name, ZfxElemFilter& filter) {
        if (attr_name != "nrm" && attr_name != "pos") {
            //supporting only @N and @P
            return;
        }

        if (auto spPrim = std::dynamic_pointer_cast<PrimitiveObject>(spObject)) {
            if (spPrim->has_attr(attr_name)) {
                auto/*std::vector<vec3f>*/& attrvecs = spPrim->attr<vec3f>(attr_name);
                assert(filter.size() == attrvecs.size());
                for (int i = 0; i < attrvecs.size(); i++) {
                    if (filter[i]) {
                        const glm::vec3& vec = get_zfxvar<glm::vec3>(val.value[i]);
                        attrvecs[i] = { vec.x, vec.y, vec.z };
                    }
                }
            }
            else {
                throw makeError<UnimplError>("the prim has no attr about normal, you can check whether the option `hasNormal` is on");
            }
        }
        else if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(spObject)) {
            if (attr_name == "pos") {
                spGeo->set_points_pos(val, filter);
            }
            else if (attr_name == "nrm") {
                spGeo->set_points_normal(val, filter);
            }
        }
    }

    void FunctionManager::commitToPrim(const std::string& attrname, const ZfxVariable& val, ZfxElemFilter& filter, ZfxContext* pContext) {
        if (pContext->runover == RunOver_Points) {
            if (attrname == "@P") {
                commitToObject(pContext->spObject, val, "pos", filter);
            }
            else if (attrname == "@ptnum") {
                throw makeError<UnimplError>("");
            }
            else if (attrname == "@N") {
                commitToObject(pContext->spObject, val, "nrm", filter);
            }
            else if (attrname == "@Cd") {
            }
            else {
            }
        }
        else if (pContext->runover == RunOver_Face) {
            
        }
        else if (pContext->runover == RunOver_Geom) {
            
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
                for (auto pos : P) {
                    res.value.push_back(glm::vec3(pos[0], pos[1], pos[2]));
                }
                return res;
            }
            if (attr_name == "ptnum")
            {
                int N = spPrim->verts->size();
                ZfxVariable res;
                res.value.resize(N);
                res.bAttr = true;
                for (int i = 0; i < N; i++)
                    res.value[i] = i;
                return res;
            }
            if (attr_name == "nrm")
            {
                if (spPrim->has_attr("nrm")) {
                    const auto& nrms = spPrim->attr<vec3f>("nrm");
                    ZfxVariable res;
                    res.bAttr = true;
                    for (auto nrm : nrms) {
                        res.value.push_back(glm::vec3(nrm[0], nrm[1], nrm[2]));
                    }
                    return res;
                }
                else {
                    throw makeError<UnimplError>("the prim has no attr about normal, you can check whether the option `hasNormal` is on");
                }
            }
        }
        else if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(spObject)) {
            if (attr_name == "pos")
            {
                const auto& P = spGeo->get_points();
                ZfxVariable res;
                res.bAttr = true;
                for (auto pos : P) {
                    res.value.push_back(glm::vec3(pos[0], pos[1], pos[2]));
                }
                return res;
            }
            if (attr_name == "ptnum")
            {
                int N = spGeo->get_point_count();
                ZfxVariable res;
                res.value.resize(N);
                res.bAttr = true;
                for (int i = 0; i < N; i++)
                    res.value[i] = i;
                return res;
            }
            if (attr_name == "nrm")
            {
                if (spGeo->has_point_attr("nrm")) {
                    ZfxVariable res;
                    res.bAttr = true;
                    res.value = spGeo->get_point_attr("nrm");
                    return res;
                }
                else {
                    throw makeError<UnimplError>("the prim has no attr about normal, you can check whether the option `hasNormal` is on");
                }
            }
        }
        else {
            return ZfxVariable();
        }
    }

    ZfxVariable FunctionManager::getAttrValue(const std::string& attrname, ZfxContext* pContext) {
        if (pContext->runover == RunOver_Points) {
            if (attrname == "@P") {
                return getAttrValue_impl(pContext->spObject, "pos");
            }
            else if (attrname == "@ptnum") {
                return getAttrValue_impl(pContext->spObject, "ptnum");
            }
            else if (attrname == "@N") {
                return getAttrValue_impl(pContext->spObject, "nrm");
            }
            else if (attrname == "@Cd") {
                return ZfxVariable();
            }
            else {
                return ZfxVariable();
            }
        }
        else if (pContext->runover == RunOver_Face) {
            return ZfxVariable();
        }
        else if (pContext->runover == RunOver_Geom) {
            return ZfxVariable();
        }
        else {
            return ZfxVariable();
        }
    }

    ZfxVariable FunctionManager::execute(std::shared_ptr<ZfxASTNode> root, ZfxElemFilter& filter, ZfxContext* pContext) {
        if (!root) {
            throw makeError<UnimplError>("Indexing Error.");
        }
        switch (root->type)
        {
            case NUMBER:
            case STRING:
            case BOOLTYPE: {
                ZfxVariable var;
                var.value.push_back(root->value);
                return var;
            }
            case ZENVAR: {
                //这里指的是取zenvar的值用于上层的计算或者输出，赋值并不会走到这里
                const std::string& varname = get_zfxvar<std::string>(root->value);
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
                case COMPVISIT: {
                    if (root->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error on NameVisit");
                    }
                    std::string component = get_zfxvar<std::string>(root->children[0]->value);
                    return get_element_by_name(var, component);
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
                case AutoDecreaseLast:  //在外面再自增/减
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
                std::shared_ptr<ZfxASTNode> zenvarNode = root->children[0];
                std::shared_ptr<ZfxASTNode> valNode = root->children[1];

                const std::string& targetvar = get_zfxvar<std::string>(zenvarNode->value);

                ZfxVariable res = execute(valNode, filter, pContext);

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
                            assert(res.value.size() <= var.value.size());
                            if (res.value.size() < var.value.size()) {
                                //如果右边的值的容器大小比当前赋值属性要小，很可能是单值，先只考虑这种情况。
                                assert(res.value.size() == 1);
                                std::fill(var.value.begin(), var.value.end(), res.value[0]);
                            }
                            else {
                                var = std::move(res);
                            }
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
                        newvar.value.push_back(0);
                        break;
                    }
                    case TYPE_INT_ARR: {
                        newvar.value.push_back(zfxintarr());
                        break;
                    }
                    case TYPE_FLOAT: {
                        newvar.value.push_back(0.f);
                        break;
                    }
                    case TYPE_FLOAT_ARR: {
                        newvar.value.push_back(zfxfloatarr());
                        break;
                    }
                    case TYPE_STRING: {
                        newvar.value.push_back("");
                        break;
                    }
                    case TYPE_STRING_ARR: {
                        newvar.value.push_back(zfxstringarr());
                        break;
                    }
                    case TYPE_VECTOR2: {
                        newvar.value.push_back(glm::vec2());
                        break;
                    }
                    case TYPE_VECTOR3: {
                        newvar.value.push_back(glm::vec3());
                        break;
                    }
                    case TYPE_VECTOR4:  newvar.value.push_back(glm::vec4()); break;
                    case TYPE_MATRIX2:  newvar.value.push_back(glm::mat2()); break;
                    case TYPE_MATRIX3:  newvar.value.push_back(glm::mat3()); break;
                    case TYPE_MATRIX4:  newvar.value.push_back(glm::mat4()); break;
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
                ZfxVariable result = eval(funcname, args, filter, pContext);
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
                std::vector<ZfxVariable> args = process_args(root, filter, pContext);
                if (args.size() != 3) {
                    throw makeError<UnimplError>("cond exp args");
                }
                auto& pCond = args[0];

                ZfxElemFilter newFilter;
                if (hasTrue(pCond, filter, newFilter)) {
                    auto pCodesExp = root->children[1];
                    return execute(pCodesExp, newFilter, pContext);
                }
                else {
                    auto pCodesExp = root->children[2];
                    return execute(pCodesExp, newFilter, pContext);
                }
            }
            case IF:{
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("if cond failed.");
                }
                auto pCondExp = root->children[0];
                //todo: self inc
                const ZfxVariable& cond = execute(pCondExp, filter, pContext);
                ZfxElemFilter newFilter;
                if (hasTrue(cond, filter, newFilter)) {
                    auto pCodesExp = root->children[1];
                    execute(pCodesExp, newFilter, pContext);
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
                ZfxElemFilter newFilter;
                while (hasTrue(cond, filter, newFilter)) {
                    //TODO: check the passed element and mark in the newFilter.
                    execute(loopContent, newFilter, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return ZfxVariable();

                    execute(forStep, newFilter, pContext);
                    cond = execute(forCond, newFilter, pContext);
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

                    for (auto eacharr : arr.value)
                    {
                        std::visit([&](auto&& val) {
                            using T = std::decay_t<decltype(val)>;
                            if constexpr (std::is_same_v<T, zfxintarr> ||
                                std::is_same_v<T, zfxfloatarr> ||
                                std::is_same_v<T, zfxstringarr>) {

                                for (int i = 0; i < val.size(); i++) {
                                    //修改变量和索引的值为i, arrtest[i];
                                    if (idxNode) {
                                        ZfxVariable zfxvar;
                                        zfxvar.value.push_back(i);
                                        assignVariable(idxName, zfxvar, pContext);
                                    }

                                    ZfxVariable zfxvar;
                                    zfxvar.value.push_back(val[i]);
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
                            else if constexpr (std::is_same_v<T, glm::vec2> ||
                                std::is_same_v<T, glm::vec3> ||
                                std::is_same_v<T, glm::vec4>) {

                                for (int i = 0; i < val.length(); i++) {
                                    //修改变量和索引的值为i, arrtest[i];
                                    if (idxNode) {
                                        ZfxVariable zfxvar;
                                        zfxvar.value.push_back(i);
                                        assignVariable(idxName, zfxvar, pContext);
                                    }

                                    ZfxVariable zfxvar;
                                    zfxvar.value.push_back(val[i]);
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
                            else {
                                throw makeError<UnimplError>("foreach error: no array type");
                            }
                        }, eacharr);
                    }
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
                ZfxElemFilter newFilter;
                while (hasTrue(cond, filter, newFilter)) {
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

                //压栈
                pushStack();
                scope_exit sp([this]() {this->popStack(); });
                ZfxVariable cond;

                ZfxElemFilter newFilter = filter;
                ZfxElemFilter newFilter2;

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
                } while (hasTrue(cond, newFilter, newFilter2));

                break;
            }
            case CODEBLOCK:{
                //多个语法树作为children的代码块
                //压栈，结束后退栈。
                pushStack();
                scope_exit sp([this]() {this->popStack(); });
                for (auto pSegment : root->children) {
                    execute(pSegment, filter, pContext);
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

    zfxvariant FunctionManager::calc(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext) {
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
            case nodeType::FUNC:
            {
                const std::string& funcname = std::get<std::string>(root->value);
                if (funcname == "ref") {
                    if (root->children.size() != 1) throw makeError<UnimplError>();
                    const std::string ref = std::get<std::string>(calc(root->children[0], pContext));
                    float res = callRef(ref, pContext);
                    return res;
                }
                else {
                    //先简单匹配调用
                    if (funcname == "sin") {
                        if (root->children.size() != 1) throw makeError<UnimplError>();
                        float val = std::get<float>(calc(root->children[0], pContext));
                        return sin(val);
                    }
                    else if (funcname == "cos") {
                        if (root->children.size() != 1) throw makeError<UnimplError>();
                        float val = std::get<float>(calc(root->children[0], pContext));
                        return cos(val);
                    }
                    else if (funcname == "sinh") {
                        if (root->children.size() != 1) throw makeError<UnimplError>();
                        float val = std::get<float>(calc(root->children[0], pContext));
                        return sinh(val);
                    }
                    else if (funcname == "cosh") {
                        if (root->children.size() != 1) throw makeError<UnimplError>();
                        float val = std::get<float>(calc(root->children[0], pContext));
                        return cosh(val);
                    }
                    else if (funcname == "rand") {
                        if (!root->children.empty()) throw makeError<UnimplError>();
                        return rand();
                    }
                    else {
                        throw makeError<UnimplError>();
                    }
                }
            }
        }
        return zfxvariant();
    }

    std::string format_variable_size(const char* fmt, std::vector<zfxvariant> args) {
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
                    else {
                        throw makeError<UnimplError>("error type on format string");
                    }
                    }, arg);
            }
        );
    }

    ZfxVariable FunctionManager::eval(const std::string& funcname, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext) {
        if (funcname == "ref") {
            if (args.size() != 1)
                throw makeError<UnimplError>("only support non-attr value when using ref");
            const std::string ref = get_zfxvar<std::string>(args[0].value[0]);
            float res = callRef(ref, pContext);
            ZfxVariable varres;
            varres.value.push_back(res);
            return varres;
        }
        else if (funcname == "log") {
            if (args.empty()) {
                throw makeError<UnimplError>("empty args on log");
            }
            const auto& formatStr = args[0];
            assert(formatStr.value.size() == 1);
            std::string formatString = get_zfxvar<std::string>(formatStr.value[0]);

            std::vector<ZfxVariable> _args = args;
            _args.erase(_args.begin());

            //有可能是： log("format", 2, @P.x, b);  //这里的_args的元素，可能是一个或多个。
            int maxSize = 1;
            for (auto& arg : _args) {
                maxSize = max(maxSize, arg.value.size());
            }

            //逐个调用输出
            if (maxSize > 1) {
                //属性或相关变量的调用
                for (int i = 0; i < maxSize; i++) {
                    if (!filter[i]) continue;
                    std::vector<zfxvariant> formatargs;
                    for (int j = 0; j < _args.size(); j++) {
                        auto& arg = _args[j];
                        assert(!arg.value.empty());
                        if (arg.value.size() < i) {
                            formatargs.push_back(arg.value[0]);
                        }
                        else {
                            formatargs.push_back(arg.value[i]);
                        }
                    }
                    std::string ret = format_variable_size(formatString.c_str(), formatargs);
                    zeno::log_only_print(ret);
                }
            }
            else {
                std::vector<zfxvariant> __args;
                for (auto __arg : _args) {
                    __args.push_back(__arg.value[0]);
                }
                std::string ret = format_variable_size(formatString.c_str(), __args);
                zeno::log_only_print(ret);
            }

            return ZfxVariable();
        }
        else {
            //先简单匹配调用
            if (funcname == "sin") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                const auto& arg = args[0];
                int N = arg.value.size();
                ZfxVariable res;
                res.value.resize(N);
                assert(N >= 1);
                if (N > 1) {
                    for (int i = 0; i < arg.value.size(); i++)
                    {
                        if (!filter[i]) continue;
                        float val = get_zfxvar<float>(arg.value[i]);
                        res.value[i] = sin(val);
                    }
                }
                else {
                    float val = get_zfxvar<float>(arg.value[0]);
                    res.value[0] = sin(val);
                }
                return res;
            }
            else if (funcname == "cos") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                const auto& arg = args[0];
                ZfxVariable res;
                res.value.resize(arg.value.size());
                for (int i = 0; i < arg.value.size(); i++)
                {
                    if (!filter[i]) continue;
                    float val = get_zfxvar<float>(arg.value[i]);
                    res.value[i] = cos(val);
                }
                return res;
            }
            else if (funcname == "sinh") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                const auto& arg = args[0];
                ZfxVariable res;
                res.value.resize(arg.value.size());
                for (int i = 0; i < arg.value.size(); i++)
                {
                    if (!filter[i]) continue;
                    float val = get_zfxvar<float>(arg.value[i]);
                    res.value[i] = sinh(val);
                }
                return res;
            }
            else if (funcname == "cosh") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                const auto& arg = args[0];
                ZfxVariable res;
                res.value.resize(arg.value.size());
                for (int i = 0; i < arg.value.size(); i++)
                {
                    if (!filter[i]) continue;
                    float val = get_zfxvar<float>(arg.value[i]);
                    res.value[i] = cosh(val);
                }
                return res;
            }
            else if (funcname == "rand") {
                if (!args.empty()) throw makeError<UnimplError>();
                ZfxVariable res;
                res.value.push_back(rand());
                return res;
            }
            else if (funcname == "addpoint") {
                if (args.size() == 1) {
                    const auto& arg = args[0];
                    if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(pContext->spObject)) {
                        //暂时只考虑一个点
                        int ptnum = spGeo->addpoint(arg.value[0]);
                        ZfxVariable res;
                        res.value.push_back(ptnum);
                        return res;
                    }
                    else {
                        throw makeError<UnimplError>();
                    }
                }
                else if (args.empty()) {
                    if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(pContext->spObject)) {
                        //暂时只考虑一个点
                        int ptnum = spGeo->addpoint();
                        ZfxVariable res;
                        res.value.push_back(ptnum);
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
            else if (funcname == "addvertex") {
                if (args.size() != 2) {
                    throw makeError<UnimplError>();
                }
                if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(pContext->spObject)) {
                    int faceid = get_zfxvar<int>(args[0].value[0]);
                    int pointid = get_zfxvar<int>(args[1].value[0]);
                    int vertid = spGeo->addvertex(faceid, pointid);
                    ZfxVariable res;
                    res.value.push_back(vertid);
                    return res;
                }
                else {
                    throw makeError<UnimplError>();
                }
            }
            else if (funcname == "removepoint") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                const auto& arg = args[0];
                int N = arg.value.size();
                if (N == 0) return ZfxVariable();
                bool bSucceed = false;

                if (N < filter.size()) {
                    assert(N == 1);
                    int currrem = get_zfxvar<int>(arg.value[0]);
                    bSucceed = removePoint(currrem, pContext);
                    if (bSucceed) {
                        //要调整filter，移除掉第currrem位置的元素
                        filter.erase(filter.begin() + currrem);
                        //所有储存在m_globalAttrCached里的属性都移除第currrem号元素，如果有ptnum，也要调整
                        afterRemovePoint(currrem);
                    }
                    else {
                        throw makeError<UnimplError>("error on removePoint");
                    }
                }
                else {
                    std::deque<int> remPoints;

                    assert(N == filter.size());
                    for (int i = 0; i < N; i++) {
                        if (!filter[i]) continue;
                        int pointnum = get_zfxvar<int>(arg.value[i]);
                        remPoints.push_back(pointnum);
                    }

                    while (!remPoints.empty())
                    {
                        int currrem = remPoints.front();
                        remPoints.pop_front();
                        bSucceed = removePoint(currrem, pContext);
                        if (bSucceed) {
                            //要调整filter，移除掉第currrem位置的元素
                            filter.erase(filter.begin() + currrem);
                            //所有储存在m_globalAttrCached里的属性都移除第currrem号元素，如果有ptnum，也要调整
                            afterRemovePoint(currrem);
                            //最后将当前所有剩下的删除点的序号再自减
                            for (auto iter = remPoints.begin(); iter != remPoints.end(); iter++) {
                                *iter -= 1;
                            }
                        }
                        else {
                            throw makeError<UnimplError>("error on removePoint");
                        }
                    }
                }
                return ZfxVariable();
            }
            else if (funcname == "removeface") {
                if (args.size() > 2 || args.empty())
                    throw makeError<UnimplError>();

                auto spGeo = std::dynamic_pointer_cast<GeometryObject>(pContext->spObject);
                const auto& arg = args[0];
                bool bIncludePoints = true;
                if (args.size() == 2) {
                    bIncludePoints = get_zfxvar<bool>(args[1].value[0]);
                }

                int N = arg.value.size();
                if (N == 0) return ZfxVariable();
                bool bSucceed = false;

                std::set<int> remfaces;
                if (N < filter.size()) {
                    assert(N == 1);
                    int currrem = get_zfxvar<int>(arg.value[0]);
                    remfaces.insert(currrem);
                }
                else {
                    assert(N == filter.size());
                    for (int i = 0; i < N; i++) {
                        if (!filter[i]) continue;
                        int pointnum = get_zfxvar<int>(arg.value[i]);
                        remfaces.insert(pointnum);
                    }
                }

                bSucceed = spGeo->remove_faces(remfaces, bIncludePoints);
                if (bSucceed) {
                    //要调整filter，移除掉第currrem位置的元素
                    removeElemsByIndice(filter, remfaces);
                    afterRemoveElements(remfaces);
                }
                else {
                    throw makeError<UnimplError>("error on removeface");
                }
                return ZfxVariable();
            }
            else {
                throw makeError<UnimplError>();
            }
        }
    }

    bool FunctionManager::removePoint(int pointnum, ZfxContext* pContext) {
        /* 删除pointnum的点，如果成功，就返回原来下一个点的pointnum(应该就是pointnum)，失败就返回-1 */
        if (auto spGeo = std::dynamic_pointer_cast<GeometryObject>(pContext->spObject)) {
            return spGeo->remove_point(pointnum);
        }
        return false;
    }

    void FunctionManager::afterRemoveElements(std::set<int> rm_indice) {
        for (auto& [name, attrVar] : m_globalAttrCached) {
            auto& attrvalues = attrVar.value;
            removeElemsByIndice(attrvalues, rm_indice);
        }
    }

    void FunctionManager::afterRemovePoint(int rempoint) {
        for (auto& [name, attrVar] : m_globalAttrCached) {
            auto& attrvalues = attrVar.value;
            if (name == "@ptnum") {
                assert(rempoint < attrvalues.size());
                for (int i = rempoint+1; i < attrvalues.size(); i++)
                    attrvalues[i] = i - 1;
                attrvalues.erase(attrvalues.begin() + rempoint);
            }
            else {
                attrvalues.erase(attrvalues.begin() + rempoint);
            }
        }
    }

    void FunctionManager::init() {
        m_funcs = {
            {"sin", 
                {"sin",
                "Return the sine of the argument",
                "float",
                {{"degree", "float"}}
                }
            },
            {"cos",
                {"cos",
                "Return the cose of the argument",
                "float",
                { {"degree", "float"}}}
            },
            {"sinh",
                {"sinh",
                "Return the hyperbolic sine of the argument",
                "float",
                { {"number", "float"}}}
            },
            {"cosh",
                {"cosh",
                "Return the hyperbolic cose of the argument",
                "float",
                { {"number", "float"}}}
            },
            {"ref",
                {"ref",
                "Return the value of reference param of node",
                "float",
                { {"path-to-param", "string"}}}
            },
            {"rand",
                {"rand",
                "Returns a pseudo-number number from 0 to 1",
                "float", {}}
            }
        };
    }

}