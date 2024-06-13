#ifndef __CORE_GLOBALVARIABLE_H__
#define __CORE_GLOBALVARIABLE_H__

#include <zeno/core/common.h>
#include <zeno/utils/helper.h>

#include <zeno/utils/api.h>
#include <memory>
#include <map>

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

    ZENO_API GVariable() { gvar = zvariant(); name = ""; gvarType = GV_UNDEFINE; }
    ZENO_API GVariable(std::string globalvarName, zvariant globalvar);
    bool operator==(const GVariable& var1) {
        ParamType tmptype;
        return var1.name == name && zeno::isEqual(var1.gvar, gvar, tmptype);
    }
};

struct OverrdeVector {
    std::vector<GVariable> vector;
    GlobalVariableType variableType;
    OverrdeVector() { variableType = GV_UNDEFINE; }
    OverrdeVector(std::vector<GVariable> v, GlobalVariableType vtype) : vector(v), variableType(vtype) {}
};

struct GlobalVariableStack {
    std::map<std::string, OverrdeVector> GlobalVariables;

    ZENO_API bool updateVariable(const GVariable& newvar);
    ZENO_API bool overrideVariable(const GVariable& var);
    void cancelOverride(std::string varname, GVariable& cancelVar);
    ZENO_API zvariant getVariable(std::string varname);
};

struct GlobalVariableOverride {
    std::shared_ptr<INode> currNode;
    GVariable gvar;
    bool overrideSuccess;

    ZENO_API GlobalVariableOverride(std::shared_ptr<INode> node, GVariable globalVariable);;
    ZENO_API ~GlobalVariableOverride();;
    ZENO_API bool updateGlobalVariable(GVariable globalVariable);
};

}

#endif