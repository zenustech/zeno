#include <zeno/core/FunctionManager.h>

namespace zeno {

    FunctionManager::FunctionManager() {
        init();
    }

    std::vector<std::string> FunctionManager::getCandidates(const std::string& prefix) const {
        std::vector<std::string> candidates;
        if (prefix.empty())
            return candidates;

        for (auto& [k, v] : m_funcs) {
            //TODO: optimize the search
            if (k.substr(0, prefix.size()) == prefix) {
                candidates.push_back(k);
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