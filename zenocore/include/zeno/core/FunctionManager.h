#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>

namespace zeno {

    class FunctionManager
    {
    public:
        FunctionManager();
        std::vector<std::string> getCandidates(const std::string& prefix) const;
        std::string getFuncTip(const std::string& funcName, bool& bExist) const;
        FUNC_INFO getFuncInfo(const std::string& funcName) const;

    private:
        void init();

        std::map<std::string, FUNC_INFO> m_funcs;
    };
}

#endif