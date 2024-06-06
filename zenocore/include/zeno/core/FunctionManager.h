#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>
#include <zeno/formula/syntax_tree.h>

namespace zeno {

    class FunctionManager
    {
    public:
        FunctionManager();
        std::vector<std::string> getCandidates(const std::string& prefix, bool bFunc) const;
        std::string getFuncTip(const std::string& funcName, bool& bExist) const;
        FUNC_INFO getFuncInfo(const std::string& funcName) const;
        zvariant calc(std::shared_ptr<ZfxASTNode> root, FuncContext* pContext);

    private:
        void init();
        float callRef(const std::string& ref, FuncContext* pContext);

        std::map<std::string, FUNC_INFO> m_funcs;
    };
}

#endif