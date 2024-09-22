#ifndef __FUNCTION_MANAGER_H__
#define __FUNCTION_MANAGER_H__

#include <vector>
#include <string>
#include <map>
#include <zeno/core/data.h>
#include <zeno/utils/api.h>
#include <zeno/formula/syntax_tree.h>
#include <stack>

namespace zeno {

    struct ZfxVariable
    {
        std::vector<zfxvariant> value;  //如果是属性变量(bAttr=true)，那这个容器的大小就是runover（点线面）的元素个数，否则就是size=1
        bool bAttr = false;     //是否与属性关联（好像没什么用）
        bool bAttrUpdated = false;      //ZfxVariable也记录属性值（比如@P, @N @ptnum等），此标记记录在zfx执行中，属性值是否修改了

        ZfxVariable() {}
        ZfxVariable(zfxvariant&& var) {
            value.emplace_back(var);
        }
    };

    using VariableTable = std::map<std::string, ZfxVariable>;
    using ZfxVarRef = VariableTable::const_iterator;
    using ZfxElemFilter = std::vector<char>;

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

        //得到所有的引用源信息，每一项是 < 节点uuid-path, 参数名 >
        std::set<std::pair<std::string, std::string>>
            getReferSources(std::shared_ptr<ZfxASTNode> root, ZfxContext* pContext);

        ZENO_API void testExp();

    private:
        void init();
        float callRef(const std::string& ref, ZfxContext* pContext);
        ZfxVariable eval(const std::string& func, const std::vector<ZfxVariable>& args, ZfxElemFilter& filter, ZfxContext* pContext);
        void pushStack();
        void popStack();
        bool hasTrue(const ZfxVariable& cond, const ZfxElemFilter& filter, ZfxElemFilter& newFilter) const;

        ZfxVariable& getVariableRef(const std::string& name, ZfxContext* pContext);
        bool declareVariable(const std::string& name);
        bool assignVariable(const std::string& name, ZfxVariable var, ZfxContext* pContext);
        void validateVar(operatorVals varType, ZfxVariable& newvar);
        ZfxVariable parseArray(std::shared_ptr<ZfxASTNode> pNode, ZfxElemFilter& filter, ZfxContext* pContext);
        ZfxVariable execute(std::shared_ptr<ZfxASTNode> root, ZfxElemFilter& filter, ZfxContext* pContext);
        std::vector<ZfxVariable> process_args(std::shared_ptr<ZfxASTNode> parent, ZfxElemFilter& filter, ZfxContext* pContext);

        ZfxVariable getAttrValue(const std::string& attrname, ZfxContext* pContext);
        void commitToPrim(const std::string& attrname, const ZfxVariable& val, ZfxElemFilter& filter, ZfxContext* pContext);
        bool removePoint(int pointnum, ZfxContext* pContext);
        void afterRemovePoint(int rempoint);
        void afterRemoveElements(std::set<int> rm_indice);

        VariableTable m_globalAttrCached;
        std::map<std::string, FUNC_INFO> m_funcs;
        std::vector<ZfxStackEnv> m_stacks;
    };
}

#endif