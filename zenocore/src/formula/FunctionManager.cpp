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

    FUNC_INFO FunctionManager::getFuncInfo(const std::string& funcName) const {
        auto iter = m_funcs.find(funcName);
        if (iter == m_funcs.end()) {
            return FUNC_INFO();
        }
        return iter->second;
    }

    float FunctionManager::callRef(const std::string& ref, ZfxContext* pContext) {
        //TODO: vec type.
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
    }

    void FunctionManager::executeZfx(std::shared_ptr<ZfxASTNode> root, ZfxContext* pCtx) {
        //debug
        //printSyntaxTree(root, pCtx->code);
        execute(root, pCtx);
    }

    template <class T>
    static T get_zfxvar(zfxvariant value) {
        return std::visit([](auto const& val) -> T {
            using V = std::decay_t<decltype(val)>;
            if constexpr (!std::is_constructible_v<T, V>) {
                throw makeError<TypeError>(typeid(T), typeid(V), "get<zfxvariant>");
            }
            else {
                return T(val);
            }
        }, value);
    }

    template<typename Operator>
    zfxvariant calc_exp(const zfxvariant& lhs, const zfxvariant& rhs, Operator method) {
        return std::visit([method](auto&& lval, auto&& rval)->zfxvariant {
            using T = std::decay_t<decltype(lval)>;
            using E = std::decay_t<decltype(rval)>;
            using Op = std::decay_t<decltype(method)>;
            if constexpr (std::is_same_v<T, int>) {
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
                else {
                    return method(lval, rval);
                }
            }
            else {
                throw UnimplError("");
            }
        }, lhs, rhs);
    }

    void FunctionManager::testExp() {

        glm::vec3 v1(1, 2, 3);
        glm::vec3 v2(1, 3, 1);

        zfxvariant vec1 = v1;
        zfxvariant vec2 = v2;
        zfxvariant vec3 = calc_exp(v1, v2, std::divides());

        glm::mat3 mat1 = glm::mat3({ {1., 0, 2.}, {2., 1., -1.}, {0, 1., 1.} });
        glm::mat3 mat2 = glm::mat3({ {1., 0, 0}, {0, -1., 1.}, {0, 0, -1.} });
        glm::mat3 mat3 = glm::mat3({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });
        glm::mat3 mat4 = glm::mat3({ {1, 0, 0}, {0, -1, 1}, {0, 0, -1} });
        glm::mat3 mm = mat3 * mat4;
        mm = mat1 * mat2;

        zfxvariant bval = calc_exp(mat1, mat2, std::equal_to());
        zfxvariant mmm = calc_exp(mat1, mat2, std::multiplies());

        //glm::mat3 mm2 = glm::dot(mat1, mat2);
    }

    static void set_array_element(zfxvariant& zfxarr, int idx, const zfxvariant& zfxvalue) {
        std::visit([idx](auto& arr, auto& value) {
            using T = std::decay_t<decltype(arr)>;
            using V = std::decay_t<decltype(value)>;
            //mat取一层索引可能就是vec...
            if constexpr ((std::is_same_v<T, zfxintarr> ||
                           std::is_same_v<T, zfxfloatarr> ||
                           std::is_same_v<T, glm::vec2> ||
                           std::is_same_v<T, glm::vec3> ||
                           std::is_same_v<T, glm::vec4>) && 
                std::is_arithmetic_v<V>) {
                arr[idx] = value;
            }
        }, zfxarr, zfxvalue);
    }

    static zfxvariant get_array_element(const zfxvariant& arr, int idx) {
        return std::visit([idx](auto&& arg) -> zfxvariant {
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
        }, arr);
    }

    static zfxvariant get_element_by_name(const zfxvariant& arr, const std::string& name) {
        int idx = -1;
        if (name == "x") idx = 0;
        else if (name == "y") idx = 1;
        else if (name == "z") idx = 2;
        else if (name == "w") idx = 3;
        else
        {
            throw makeError<UnimplError>("Indexing Exceed");
        }
        return get_array_element(arr, idx);
    }

    static void set_element_by_name(const zfxvariant& arr, const std::string& name, const zfxvariant& value) {
        //TODO
    }

    static void selfIncOrDec(zfxvariant& var, bool bInc) {
        std::visit([bInc](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                bInc ? (T)arg++ : (T)arg--;
            }
            else {
                throw makeError<UnimplError>("Type Error");
            }
        }, var);
    }

    std::vector<zfxvariant> FunctionManager::process_args(std::shared_ptr<ZfxASTNode> parent, ZfxContext* pContext) {
        std::vector<zfxvariant> args;
        for (auto pChild : parent->children) {
            zfxvariant argval = execute(pChild, pContext);
            args.push_back(argval);
        }
        return args;
    }

    void FunctionManager::pushStack() {
        m_variables.push_back(VariableTable());
    }

    void FunctionManager::popStack() {
        m_variables.pop_back();
    }

    zfxvariant& FunctionManager::getVariableRef(const std::string& name) {
        for (auto iter = m_variables.rbegin(); iter != m_variables.rend(); iter++) {
            auto iter_ = iter->find(name);
            if (iter_ != iter->end()) {
                return iter->at(name);
            }
        }
        throw makeError<KeyError>(name, "variable `" + name + "` not founded");
    }

    zfxvariant FunctionManager::getVariable(const std::string& name) const {
        for (auto iter = m_variables.rbegin(); iter != m_variables.rend(); iter++) {
            auto iter_ = iter->find(name);
            if (iter_ != iter->end()) {
                return iter_->second;
            }
        }
        throw makeError<KeyError>(name, "variable `" + name + "` not founded");
    }

    bool FunctionManager::declareVariable(const std::string& name, zfxvariant var) {
        if (m_variables.empty()) {
            return false;
        }
        auto iterCurrentStack = m_variables.rbegin();
        if (iterCurrentStack->find(name) != iterCurrentStack->end()) {
            return false;
        }
        iterCurrentStack->insert(std::make_pair(name, var));
        return true;
    }

    bool FunctionManager::assignVariable(const std::string& name, zfxvariant var) {
        if (m_variables.empty()) {
            return false;
        }
        //如果有多个重名定义，直接找最浅的栈对应的变量值。
        auto iterCurrentStack = m_variables.rbegin();
        auto iterVar = iterCurrentStack->find(name);
        if (iterVar == iterCurrentStack->end()) {
            return false;
        }
        iterVar->second = var;
        return true;
    }

    void FunctionManager::validateVar(operatorVals vartype, zfxvariant& newvar) {
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

    zfxvariant FunctionManager::parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxContext* pContext) {
        std::vector<zfxvariant> args = process_args(pNode, pContext);
        //由于不清楚类型，所以先根据值类型构造一个zfxvariant返回出去，再由外部类型定义/赋值来处理
        //由于zfxvariant并没有多维数组，因此遇到多维数组，直接构造矩阵（如果构造失败，那就throw）
        if (args.empty()) {
            //直接返回默认的
            return zfxvariant();
        }

        zfxvariant current;
        operatorVals dataType = UNDEFINE_OP;
        for (int idx = 0; idx < args.size(); idx++) {
            auto& arg = args[idx];
            if (std::holds_alternative<int>(arg)) {
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_FLOAT_ARR;
                    current = zfxfloatarr();
                }
                if (dataType == TYPE_FLOAT_ARR) {
                    auto& arr = std::get<zfxfloatarr>(current);
                    arr.push_back(std::get<int>(arg));
                }
                else {
                    throw makeError<UnimplError>("data type inconsistent");
                }
            }
            else if (std::holds_alternative<float>(arg)) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_FLOAT_ARR) {
                    throw makeError<UnimplError>("data type inconsistent");
                }
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_FLOAT_ARR;
                    current = zfxfloatarr();
                }
                if (dataType == TYPE_FLOAT_ARR) {
                    auto& arr = std::get<zfxfloatarr>(current);
                    arr.push_back(std::get<float>(arg));
                }
            }
            else if (std::holds_alternative<std::string>(arg)) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_STRING_ARR) {
                    throw makeError<UnimplError>("data type inconsistent");
                }
                if (dataType == UNDEFINE_OP) {
                    dataType = TYPE_STRING_ARR;
                    current = zfxstringarr();
                }
                if (dataType == TYPE_STRING_ARR) {
                    auto& arr = std::get<zfxstringarr>(current);
                    arr.push_back(std::get<std::string>(arg));
                }
            }
            //不考虑intarr，因为glm的vector/matrix都是储存float
            else if (std::holds_alternative<zfxfloatarr>(arg)) {
                if (dataType != UNDEFINE_OP && dataType != TYPE_MATRIX2 && dataType != TYPE_MATRIX3 && dataType != TYPE_MATRIX4) {
                    throw makeError<UnimplError>("data type inconsistent");
                }

                auto& arr = std::get<zfxfloatarr>(arg);

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
        return current;
    }

    zfxvariant FunctionManager::execute(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext) {
        if (!root) {
            throw makeError<UnimplError>("Indexing Error.");
        }
        switch (root->type)
        {
            case NUMBER:
            case STRING:
            case BOOLTYPE: return root->value;
            case ZENVAR: {
                switch (root->opVal) {
                case Indexing: {
                    if (root->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error.");
                    }
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    int idx = get_zfxvar<int>(execute(root->children[0], pContext));
                    const zfxvariant& var = getVariable(varname);
                    zfxvariant elemvar = get_array_element(var, idx);
                    return elemvar;
                }
                case AttrMark: {
                    std::string attrname = get_zfxvar<std::string>(root->value);
                    //@P @N
                    //先支持常见的:
                    if (attrname == "P" || attrname == "N") {
                        auto& attrval = pContext->spObject->attr<zeno::vec3f>(attrname);
                        auto p = zeno::vec_to_other<glm::vec3>(attrval[0]);
                        zfxvariant var = p;
                    }
                    else {
                        throw makeError<UnimplError>("Indexing Error.");
                    }
                }
                case COMPVISIT: {
                    if (root->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error on NameVisit");
                    }
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    std::string component = get_zfxvar<std::string>(root->children[0]->value);
                    const zfxvariant& var = getVariable(varname);
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
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    zfxvariant& var = getVariable(varname);
                    selfIncOrDec(var, true);
                    return var;
                }
                case AutoDecreaseFirst: {
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    zfxvariant& var = getVariable(varname);
                    selfIncOrDec(var, false);
                    return var;
                }
                case AutoIncreaseLast:
                case AutoDecreaseLast:  //在外面再自增/减
                {
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    zfxvariant var = getVariable(varname);
                    zfxvariant updatevar = var;
                    selfIncOrDec(updatevar, AutoIncreaseLast == root->opVal);
                    assignVariable(varname, updatevar);
                    return var;
                }
                default: {
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    const zfxvariant& var = getVariable(varname);
                    return var;
                }
                }
            }
            case FUNC: {
                //函数
                std::vector<zfxvariant> args = process_args(root, pContext);
                const std::string& funcname = get_zfxvar<std::string>(root->value);
                zfxvariant result = eval(funcname, args, pContext);
                return result;
            }
            case FOUROPERATIONS: {
                //四则运算+ - * / %
                std::vector<zfxvariant> args = process_args(root, pContext);
                if (args.size() != 2) {
                    throw makeError<UnimplError>("op args");
                }
                switch (root->opVal) {
                case PLUS:  return calc_exp(args[0], args[1], std::plus());
                case MINUS: return calc_exp(args[0], args[1], std::minus());
                case MUL:   return calc_exp(args[0], args[1], std::multiplies());
                case DIV:   return calc_exp(args[0], args[1], std::divides());
                default:
                    throw makeError<UnimplError>("op error");
                }
            }
            case COMPOP: {
                //操作符
                std::vector<zfxvariant> args = process_args(root, pContext);
                if (args.size() != 2) {
                    throw makeError<UnimplError>("compare op args");
                }
                switch (root->opVal) {
                case Less:          return calc_exp(args[0], args[1], std::less());
                case LessEqual:     return calc_exp(args[0], args[1], std::less_equal());
                case Greater:       return calc_exp(args[0], args[1], std::greater());
                case GreaterEqual:  return calc_exp(args[0], args[1], std::greater_equal());
                case Equal:         return calc_exp(args[0], args[1], std::equal_to());
                case NotEqual:      return calc_exp(args[0], args[1], std::not_equal_to());
                default:
                    throw makeError<UnimplError>("compare op error");
                }
            }             
            case CONDEXP: {
                //条件表达式
                std::vector<zfxvariant> args = process_args(root, pContext);
                if (args.size() != 3) {
                    throw makeError<UnimplError>("cond exp args");
                }
                return std::visit([args](auto&& val) -> zfxvariant {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                        int condval = (int)val;
                        if (condval) {
                            return args[1];
                        }
                        else {
                            return args[2];
                        }
                    }
                    else {
                        throw makeError<UnimplError>("condexp type error");
                    }
                }, args[0]);
            }
            case ARRAY:{
                return parseArray(root, pContext);
            }
            case PLACEHOLDER:{
                return zfxvariant();
            }
            case DECLARE:{
                //变量定义
                int nChildren = root->children.size();
                if (nChildren != 2 && nChildren != 3) {
                    throw makeError<UnimplError>("args of DECLARE");
                }
                std::shared_ptr<ZfxASTNode> typeNode = root->children[0];
                std::shared_ptr<ZfxASTNode> nameNode = root->children[1];
                zfxvariant newvar;
                bool bOnlyDeclare = nChildren == 2;
                operatorVals vartype = typeNode->opVal;
                if (bOnlyDeclare) {
                    switch (vartype)
                    {
                    case TYPE_INT:      newvar = 0; break;
                    case TYPE_INT_ARR:  newvar = zfxintarr();   break;
                    case TYPE_FLOAT:    newvar = 0.f;   break;
                    case TYPE_FLOAT_ARR:    newvar = zfxfloatarr(); break;
                    case TYPE_STRING:       newvar = "";    break;
                    case TYPE_STRING_ARR:   newvar = zfxstringarr(); break;
                    case TYPE_VECTOR2:  newvar = glm::vec2(); break;
                    case TYPE_VECTOR3:  newvar = glm::vec3(); break;
                    case TYPE_VECTOR4:  newvar = glm::vec4(); break;
                    case TYPE_MATRIX2:  newvar = glm::mat2(); break;
                    case TYPE_MATRIX3:  newvar = glm::mat3(); break;
                    case TYPE_MATRIX4:  newvar = glm::mat4(); break;
                    }
                }
                else {
                    std::shared_ptr<ZfxASTNode> valueNode = root->children[2];
                    newvar = execute(valueNode, pContext);
                }

                std::string varname = get_zfxvar<std::string>(nameNode->value);
                //暂时不考虑自定义结构体
                bool bret = declareVariable(varname, newvar);
                if (!bret) {
                    throw makeError<UnimplError>("assign variable failed.");
                }

                validateVar(vartype, newvar);

                bret = assignVariable(varname, newvar);
                if (!bret) {
                    throw makeError<UnimplError>("assign variable failed.");
                }
                break;
            }
            case ASSIGNMENT:{
                //赋值
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("assign variable failed.");
                }
                std::shared_ptr<ZfxASTNode> zenvarNode = root->children[0];
                std::shared_ptr<ZfxASTNode> valNode = root->children[1];

                const std::string& targetvar = get_zfxvar<std::string>(zenvarNode->value);

                zfxvariant res = execute(valNode, pContext);

                //直接解析变量
                switch (zenvarNode->opVal) {
                case Indexing: {
                    if (valNode->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error.");
                    }
                    const std::string& varname = get_zfxvar<std::string>(valNode->value);
                    int idx = get_zfxvar<int>(execute(valNode->children[0], pContext));
                    zfxvariant& var = getVariableRef(varname);
                    set_array_element(var, idx, res);
                    return zfxvariant();  //无需返回什么具体的值
                }
                case AttrMark: {
                    std::string attrname = get_zfxvar<std::string>(valNode->value);
                    //@P @N
                    //先支持常见的:
                    if (attrname == "P" || attrname == "N") {
                        auto& attrval = pContext->spObject->attr<zeno::vec3f>(attrname);
                        auto p = zeno::vec_to_other<glm::vec3>(attrval[0]);
                        zfxvariant var = p;
                        //set attr on prim.
                    }
                    else {
                        throw makeError<UnimplError>("Indexing Error.");
                    }
                }
                case COMPVISIT: {
                    if (valNode->children.size() != 1) {
                        throw makeError<UnimplError>("Indexing Error on NameVisit");
                    }
                    const std::string& varname = get_zfxvar<std::string>(valNode->value);
                    std::string component = get_zfxvar<std::string>(valNode->children[0]->value);
                    zfxvariant& var = getVariableRef(varname);
                    set_element_by_name(var, varname, res);
                    return zfxvariant();
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
                    assignVariable(targetvar, res);
                    return zfxvariant();
                }
                }
                break;
            }
            case IF:{
                if (root->children.size() != 2) {
                    throw makeError<UnimplError>("if cond failed.");
                }
                auto pCondExp = root->children[0];
                //todo: self inc
                int cond = get_zfxvar<int>(execute(pCondExp, pContext));
                if (cond) {
                    auto pCodesExp = root->children[1];
                    execute(pCodesExp, pContext);
                }
                return zfxvariant();
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
                    execute(forBegin, pContext);
                    break;
                }
                case ASSIGNMENT:
                {
                    execute(forBegin, pContext);
                    break;
                }
                case PLACEHOLDER:
                default:
                    //填其他东西没有意义，甚至可能导致解析出错。
                    execute(forBegin, pContext);
                    break;
                }

                int cond = get_zfxvar<int>(execute(forCond, pContext));
                while (cond) {
                    execute(loopContent, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return zfxvariant();

                    execute(forStep, pContext);
                    cond = get_zfxvar<int>(execute(forCond, pContext));
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

                    zfxvariant arr = execute(arrNode, pContext);

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

                    std::visit([&](auto&& val) {
                        using T = std::decay_t<decltype(val)>;
                        if constexpr (std::is_same_v<T, zfxintarr> ||
                            std::is_same_v<T, zfxfloatarr> ||
                            std::is_same_v<T, zfxstringarr>) {

                            for (int i = 0; i < val.size(); i++) {
                                //修改变量和索引的值为i, arrtest[i];
                                if (idxNode) {
                                    assignVariable(idxName, i);
                                }
                                assignVariable(varName, val[i]);

                                //修改定义后，再次运行code
                                execute(codeSeg, pContext);

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
                                    assignVariable(idxName, i);
                                }
                                assignVariable(varName, val[i]);

                                //修改定义后，再次运行code
                                execute(codeSeg, pContext);

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
                    }, arr);
                    return zfxvariant();
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

                int cond = get_zfxvar<int>(execute(forCond, pContext));
                while (cond) {
                    execute(loopContent, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return zfxvariant();

                    cond = get_zfxvar<int>(execute(forCond, pContext));
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
                int cond = 0;

                do {
                    execute(loopContent, pContext);     //CodeBlock里面可能会多压栈一次，没关系，变量都是看得到的

                    if (pContext->jumpFlag == JUMP_BREAK)
                        break;
                    if (pContext->jumpFlag == JUMP_CONTINUE)
                        continue;
                    if (pContext->jumpFlag == JUMP_RETURN)
                        return zfxvariant();
                    int cond = get_zfxvar<int>(execute(forCond, pContext));
                } while (cond);

                break;
            }
            case CODEBLOCK:{
                //多个语法树作为children的代码块
                //压栈，结束后退栈。
                pushStack();
                scope_exit sp([this]() {this->popStack(); });
                for (auto pSegment : root->children) {
                    execute(pSegment, pContext);
                    if (pContext->jumpFlag == JUMP_BREAK ||
                        pContext->jumpFlag == JUMP_CONTINUE ||
                        pContext->jumpFlag == JUMP_RETURN) {
                        return zfxvariant();
                    }
                }
            }
            case JUMP:{
                pContext->jumpFlag = root->opVal;
                return zfxvariant();
            }
            case VARIABLETYPE:{
                //变量类型，比如int vector3 float string等
            }
            default: {
                break;
            }
        }
        return zfxvariant();
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

    zfxvariant FunctionManager::eval(const std::string& funcname, const std::vector<zfxvariant>& args, ZfxContext* pContext) {
        if (funcname == "ref") {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const std::string ref = get_zfxvar<std::string>(args[0]);
            float res = callRef(ref, pContext);
            return res;
        }
        else if (funcname == "log") {
            if (args.empty()) {
                throw makeError<UnimplError>("empty args on log");
            }
            std::string formatString = get_zfxvar<std::string>(args[0]);

            std::vector<zfxvariant> _args = args;
            _args.erase(_args.begin());

            std::string ret = format_variable_size(formatString.c_str(), _args);
            zeno::log_only_print(ret);
            //pContext->printContent += ret;
        }
        else {
            //先简单匹配调用
            if (funcname == "sin") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                float val = get_zfxvar<float>(args[0]);
                return sin(val);
            }
            else if (funcname == "cos") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                float val = get_zfxvar<float>(args[0]);
                return cos(val);
            }
            else if (funcname == "sinh") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                float val = get_zfxvar<float>(args[0]);
                return sinh(val);
            }
            else if (funcname == "cosh") {
                if (args.size() != 1)
                    throw makeError<UnimplError>();
                float val = get_zfxvar<float>(args[0]);
                return cosh(val);
            }
            else if (funcname == "rand") {
                if (!args.empty()) throw makeError<UnimplError>();
                return rand();
            }
            else {
                throw makeError<UnimplError>();
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