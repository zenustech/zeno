#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>
#include <zeno/utils/api.h>
#include <zeno/formula/syntax_tree.h>
#include <stack>
#include <regex>

namespace zeno {

    struct ZfxStackEnv
    {
        VariableTable table;
        bool bAttrAddOrRemoved = false;
        size_t indexToCurrentElem = 0;
    };

    class FunctionManager
    {
    public:
        FunctionManager();
        std::vector<std::string> getCandidates(const std::string& prefix, bool bFunc) const;
        std::string getFuncTip(const std::string& funcName, bool& bExist) const;
        ZENO_API FUNC_INFO getFuncInfo(const std::string& funcName) const;
        void executeZfx(std::shared_ptr<ZfxASTNode> root, ZfxContext* ctx);
        zfxvariant calc(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);

        ZfxVariable execute(std::shared_ptr<ZfxASTNode> root, ZfxElemFilter& filter, ZfxContext* pContext);

        //得到所有的引用源信息，每一项是 < 节点uuid-path, 参数名 >
        std::set<std::pair<std::string, std::string>>
            getReferSources(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);

        ZENO_API void testExp();

        static const std::regex refPattern;
        static const std::regex refStrPattern;

    private:
        ZfxVariable eval(const std::string& func, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext);
        void pushStack();
        void popStack();
        bool hasTrue(const ZfxVariable& cond, const ZfxElemFilter& filter, ZfxElemFilter& ifFilter, ZfxElemFilter& elseFilter) const;
        bool hasGeomTopoQueryModify(std::shared_ptr<ZfxASTNode> pNode) const;

        ZfxVariable& getVariableRef(const std::string& name, ZfxContext* pContext);
        bool declareVariable(const std::string& name);
        bool assignVariable(const std::string& name, ZfxVariable var, ZfxContext* pContext);
        void validateVar(operatorVals varType, ZfxVariable& newvar);
        ZfxVariable parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxElemFilter& filter, ZfxContext* pContext);

        std::vector<ZfxVariable> process_args(std::shared_ptr<ZfxASTNode> parent, ZfxElemFilter& filter, ZfxContext* pContext);

        ZfxVariable getAttrValue(const std::string& attrname, ZfxContext* pContext, char channel = 0);
        void setAttrValue(const std::string& attrname, const std::string& channel, const ZfxVariable& var, operatorVals opVal, ZfxElemFilter& filter, ZfxContext* pContext);

        ZfxVariable trunkVariable(ZfxVariable origin, const ZfxElemFilter& filter);
        void commitToPrim(const std::string& attrname, const ZfxVariable& val, ZfxElemFilter& filter, ZfxContext* pContext);

        VariableTable m_globalAttrCached;
        std::vector<ZfxStackEnv> m_stacks;
    };
}

#endif