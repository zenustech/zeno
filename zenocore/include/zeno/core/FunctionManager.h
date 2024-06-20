#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>
#include <zeno/formula/syntax_tree.h>
#include <stack>

namespace zeno {

    class FunctionManager
    {
        using VariableTable = std::map<std::string, zfxvariant>;
        using ZfxVarRef = VariableTable::const_iterator;

    public:
        FunctionManager();
        std::vector<std::string> getCandidates(const std::string& prefix, bool bFunc) const;
        std::string getFuncTip(const std::string& funcName, bool& bExist) const;
        FUNC_INFO getFuncInfo(const std::string& funcName) const;
        void executeZfx(std::shared_ptr<ZfxASTNode> root, ZfxContext* ctx);
        zfxvariant calc(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);
        ZENO_API void testExp();

    private:
        void init();
        float callRef(const std::string& ref, ZfxContext* pContext);
        zfxvariant eval(const std::string& func, const std::vector<zfxvariant>& args, ZfxContext* pContext);
        void pushStack();
        void popStack();

        zfxvariant getVariable(const std::string& name) const;
        zfxvariant& getVariableRef(const std::string& name);
        bool declareVariable(const std::string& name, zfxvariant var = zfxvariant());
        bool assignVariable(const std::string& name, zfxvariant var);
        void validateVar(operatorVals varType, zfxvariant& newvar);
        zfxvariant parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxContext* pContext);
        zfxvariant execute(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);
        std::vector<zfxvariant> process_args(std::shared_ptr<ZfxASTNode> parent, ZfxContext* pContext);

        std::map<std::string, FUNC_INFO> m_funcs;
        std::vector<VariableTable> m_variables;
    };
}

#endif