#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>
#include <zeno/formula/syntax_tree.h>
#include <stack>

namespace zeno {

    struct ZfxVariable
    {
        zfxvariant value;
        std::set<std::string> attachAttrs;
        std::vector<std::shared_ptr<ZfxASTNode>> assignStmts;
    };

    class FunctionManager
    {
        using VariableTable = std::map<std::string, ZfxVariable>;
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
        void updateGeomAttr(const std::string& attrname, zfxvariant value, operatorVals op, zfxvariant opval, ZfxContext* pContext);

        zfxvariant getVariable(const std::string& name) const;
        ZfxVariable& getVariableRef(const std::string& name);
        bool declareVariable(const std::string& name, zfxvariant var = zfxvariant());
        bool declareVariable(const std::string& name, ZfxVariable var);
        bool assignVariable(const std::string& name, ZfxVariable var);
        void validateVar(operatorVals varType, zfxvariant& newvar);
        zfxvariant parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxContext* pContext);
        zfxvariant execute(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);
        std::set<std::string> parsingAttr(std::shared_ptr<ZfxASTNode> root, std::shared_ptr<ZfxASTNode> spOverrideStmt, ZfxContext* pContext);
        void removeAttrvarDeclareAssign(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);
        void embeddingForeach(std::shared_ptr<ZfxASTNode> root, std::shared_ptr<ZfxASTNode> spOverrideStmt, ZfxContext* pContext);
        void getDependingVariables(const std::string& assignedVar, std::set<std::string>& vars);
        std::vector<zfxvariant> process_args(std::shared_ptr<ZfxASTNode> parent, ZfxContext* pContext);
        bool removeIrrelevantCode(std::shared_ptr<ZfxASTNode> root, int currentExecId, const std::set<std::string>& allDepvars, std::set<std::string>& allFindAttrs);
        bool isEvalFunction(const std::string& funcname);

        std::map<std::string, FUNC_INFO> m_funcs;
        std::vector<VariableTable> m_variables;
    };
}

#endif