#include <zeno/core/FunctionManager.h>
#include <zeno/core/ReferManager.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/Error.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/log.h>
#include <zeno/utils/helper.h>
#include <regex>


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

    float FunctionManager::callRef(const std::string& ref, FuncContext* pContext) {
        std::string fullPath, graphAbsPath;

        if (ref.empty()) {
            throw makeError<UnimplError>();
        }

        graphAbsPath = pContext->nodePath.substr(0, pContext->nodePath.find_last_of('/'));

        if (ref.front() == '/') {
            fullPath = ref;
        } else {
            fullPath = graphAbsPath + "/" + ref;
        }

        int idx = fullPath.find_last_of('/');
        if (idx == std::string::npos) {
            throw makeError<UnimplError>();
        }

        std::string nodePath = fullPath.substr(idx + 1);

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

    zvariant FunctionManager::calc(std::shared_ptr<ZfxASTNode> root, FuncContext* pContext) {
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
                    return zvariant();
                }
                else if (var == "T") {
                    //TODO
                    return zvariant();
                }
            }
            case nodeType::FOUROPERATIONS:
            {
                if (root->children.size() != 2)
                {
                    throw makeError<UnimplError>();
                }
                zvariant lhs = calc(root->children[0], pContext);
                zvariant rhs = calc(root->children[1], pContext);

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
                    return zvariant();
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
        return zvariant();
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