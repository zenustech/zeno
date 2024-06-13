#include <zeno/core/GlobalVariable.h>
#include <zeno/core/INode.h>

namespace zeno {

    ZENO_API bool GlobalVariableStack::updateVariable(const GVariable& newvar)
    {
        if (newvar.name.empty()) {
            return false;
        }
        GVariable oldvar;
        cancelOverride(newvar.name, oldvar);    //记录oldvar覆盖失败时取消cancel
        if (overrideVariable(newvar)) {
            return true;
        }
        else {
            overrideVariable(oldvar);
        }
        return false;
    }

    ZENO_API bool GlobalVariableStack::overrideVariable(const GVariable& var)
    {
        if (var.name.empty()) {
            return false;
        }
        auto it = GlobalVariables.find(var.name);
        if (it == GlobalVariables.end()) {
            GlobalVariables.insert(std::make_pair(var.name, OverrdeVector(std::vector<GVariable>({var}), var.gvarType) ));
            return true;
        }
        else {
            if (it->second.variableType == var.gvarType) {
                it->second.vector.push_back(var);
                return true;
            }
            else {
                //override时类型不一致；
            }
        }
        return false;
    }

    void GlobalVariableStack::cancelOverride(std::string varname, GVariable& oldlVar)
    {
        if (varname.empty()) {
            return;
        }
        auto it = GlobalVariables.find(varname);
        if (it != GlobalVariables.end()) {
            if (!it->second.vector.empty()) {
                oldlVar = it->second.vector.back();
                it->second.vector.pop_back();
            }
        } 
    }

    ZENO_API zvariant GlobalVariableStack::getVariable(std::string varname)
    {
        auto it = GlobalVariables.find(varname);
        if (it != GlobalVariables.end()) {
            if (!it->second.vector.empty()) {
                return it->second.vector.back().gvar;
            }
        }
        return zvariant();
    }

    GlobalVariableOverride::GlobalVariableOverride(std::shared_ptr<INode> node, GVariable globalVariable) : currNode(node), gvar(globalVariable)
    {
        overrideSuccess = zeno::getSession().globalVariableStack->overrideVariable(gvar);
        if (overrideSuccess) {
            currNode->propagateDirty(gvar);
        }
    }

    GlobalVariableOverride::~GlobalVariableOverride()
    {
        if (overrideSuccess) {
            GVariable oldvar;
            zeno::getSession().globalVariableStack->cancelOverride(gvar.name, oldvar);
        }
    }

    bool GlobalVariableOverride::updateGlobalVariable(GVariable globalVariable)
    {
        if (zeno::getSession().globalVariableStack->updateVariable(globalVariable)) {
            currNode->propagateDirty(globalVariable);
            return true;
        }
        return false;
    }

    ZENO_API GVariable::GVariable(std::string globalvarName, zvariant globalvar) : name(globalvarName), gvar(globalvar)
    {
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>) {
            gvarType = GV_INT;
        }
        else if constexpr (std::is_same_v<T, float>) {
            gvarType = GV_FLOAT;
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            gvarType = GV_STRING;
        }
        else if constexpr (std::is_same_v<T, zeno::vec2i>) {
            gvarType = GV_VEC2I;
        }
        else if constexpr (std::is_same_v<T, zeno::vec2f>) {
            gvarType = GV_VEC2F;
        }
        else if constexpr (std::is_same_v<T, zeno::vec3i>) {
            gvarType = GV_VEC3I;
        }
        else if constexpr (std::is_same_v<T, zeno::vec3f>) {
            gvarType = GV_VEC3F;
        }
        else if constexpr (std::is_same_v<T, zeno::vec4i>) {
            gvarType = GV_VEC4I;
        }
        else if constexpr (std::is_same_v<T, zeno::vec4f>) {
            gvarType = GV_VEC4F;
        }
        else if constexpr (std::is_same_v<T, zeno::vec2s>) {
            gvarType = GV_VEC2S;
        }
        else if constexpr (std::is_same_v<T, zeno::vec3s>) {
            gvarType = GV_VEC3S;
        }
        else if constexpr (std::is_same_v<T, zeno::vec4s>) {
            gvarType = GV_VEC4S;
        }
        else {
            gvarType = GV_UNDEFINE;
        }
        }, gvar);
    }

}