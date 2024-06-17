#ifndef __CORE_GLOBALVARIABLE_H__
#define __CORE_GLOBALVARIABLE_H__

#include <zeno/core/common.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/helper.h>

#include <zeno/utils/api.h>
#include <memory>
#include <map>
#include <stack>

namespace zeno {

class INode;

enum GlobalVariableType {   //种类和zvariant一致
    GV_UNDEFINE = 0,
    GV_INT,
    GV_VEC2I,
    GV_VEC3I,
    GV_VEC4I,
    GV_FLOAT,
    GV_VEC2F,
    GV_VEC3F,
    GV_VEC4F,
    GV_VEC2S,
    GV_VEC3S,
    GV_VEC4S,
    GV_STRING
};

struct GVariable {
    zvariant gvar;
    std::string name;
    GlobalVariableType gvarType;

    GVariable() { gvar = zvariant(); name = ""; gvarType = GV_UNDEFINE; }
    GVariable(std::string globalvarName, zvariant globalvar);
    bool operator==(const GVariable& var1) {
        ParamType tmptype;
        return var1.name == name && zeno::isEqual(var1.gvar, gvar, tmptype);
    }
};

struct OverrdeVector {
    std::stack<GVariable> stack;
    GlobalVariableType variableType;
    OverrdeVector() { variableType = GV_UNDEFINE; }
    OverrdeVector(std::stack<GVariable> v, GlobalVariableType vtype) : stack(v), variableType(vtype) {}
};

struct GlobalVariableStack {
    std::map<std::string, OverrdeVector> GlobalVariables;

    bool updateVariable(const GVariable& newvar);
    bool overrideVariable(const GVariable& var);
    void cancelOverride(std::string varname, GVariable& cancelVar);
    zvariant getVariable(std::string varname);
};

struct GlobalVariableManager
{
public:
    //查询上游dependType类型节点并传播dirty
    void propagateDirty(std::weak_ptr<INode> wpCurrNode, GVariable globalvar);
    void getUpstreamNodes(std::shared_ptr<INode> spCurrNode, std::set<ObjPath>& depNodes, std::set<ObjPath>& upstreams, std::string outParamName = "");
    void mark_dirty_by_dependNodes(std::shared_ptr<INode> spCurrNode, bool bOn, std::set<ObjPath> nodesRange, std::string inParamName = "");
    //nodepath的节点不在依赖某个全局变量
    void removeDependGlobalVaraible(ObjPath nodepath, std::string name);
    //标记nodepath的节点依赖某个全局变量
    void addDependGlobalVaraible(ObjPath nodepath, std::string name, GlobalVariableType type);

    bool updateVariable(const GVariable& newvar);
    bool overrideVariable(const GVariable& var);
    void cancelOverride(std::string varname, GVariable& cancelVar);
    zvariant getVariable(std::string varname);

    GlobalVariableStack globalVariableStack;

    std::map<ObjPath, std::map<std::string, GlobalVariableType>> globalVariablesNameTypeMap;  //存储节点依赖哪些全局变量<节点path<变量名称，变量类型>>
};

struct GlobalVariableOverride {
    std::weak_ptr<INode> currNode;
    GVariable gvar;
    bool overrideSuccess;

    ZENO_API GlobalVariableOverride(std::weak_ptr<INode> node, std::string gvarName, zvariant var);;
    ZENO_API ~GlobalVariableOverride();;
    ZENO_API bool updateGlobalVariable(GVariable globalVariable);
};

}

#endif