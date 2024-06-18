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

    void FunctionManager::executeZfx(std::shared_ptr<ZfxASTNode> root, const ZfxContext& ctx) {
        //debug
        printSyntaxTree(root, ctx.code);
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

    static zfxvariant get_element(const zfxvariant& arr, int idx) {
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
        return get_element(arr, idx);
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
            if (pChild->type == ZENVAR &&
                (pChild->opVal == AutoIncreaseLast || pChild->opVal == AutoDecreaseLast)) {

            }
        }
        return args;
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
                    zfxvariant elemvar = get_element(var, idx);
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
                default: {
                    const std::string& varname = get_zfxvar<std::string>(root->value);
                    zfxvariant& var = getVariable(varname);
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
                if (args.size() == 2) {
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
            case COMPOP:{
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
            case CONDEXP:{
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
                std::vector<zfxvariant> args = process_args(root, pContext);
                //由于不清楚类型，所以先根据值类型构造一个zfxvariant返回出去，再由外部类型定义/赋值来处理
                if (args.empty()) {
                    //直接返回默认的
                    return zfxvariant();
                }
                else {
                    std::visit([args](auto&& firstElem) -> zfxvariant {
                        using T = std::decay_t<decltype(firstElem)>;
                        if constexpr (std::is_same_v<T, int>) {
                            /* { 3, 4, 5} */
                            zfxintarr ret;
                            for (auto arg : args) {
                                if (!std::holds_alternative<int>(arg))
                                    throw makeError<UnimplError>("type of array element");
                                //ret.push_back((int)arg);
                            }
                        }
                        else if constexpr (std::is_same_v<T, float>) {
                            zfxfloatarr ret;
                            for (auto arg : args) {
                                if (!std::holds_alternative<float>(arg))
                                    throw makeError<UnimplError>("type of array element");
                                //ret.push_back((int)arg);
                            }
                        }
                        else {
                            throw makeError<UnimplError>("ARRAY type error");
                        }
                    }, args[0]);
                }
            }
            case MATRIX:{
                std::vector<zfxvariant> args = process_args(root, pContext);
                throw makeError<UnimplError>("Matrix");
            }      
            case PLACEHOLDER:{
                throw makeError<UnimplError>("placehoder occurs");
            }
            case DECLARE:{
                //变量定义
            }            
            case ASSIGNMENT:{
                //赋值
            }
            case IF:{
        
            }
            case FOR:{
        
            }
            case FOREACH:{
        
            }
            case WHILE:{
        
            }
            case DOWHILE:{
        
            }
            case CODEBLOCK:{
                //多个语法树作为children的代码块
            }
            case JUMP:{
        
            }
            case VARIABLETYPE:{
                //变量类型，比如int vector3 float string等
            }
            default: {

            }
        }
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

    zfxvariant FunctionManager::eval(const std::string& funcname, const std::vector<zfxvariant>& args, ZfxContext* pContext) {
        if (funcname == "ref") {
            if (args.size() != 1)
                throw makeError<UnimplError>();
            const std::string ref = get_zfxvar<std::string>(args[0]);
            float res = callRef(ref, pContext);
            return res;
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