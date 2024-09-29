#include <zeno/core/GlobalVariable.h>
#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>
#include "reflect/metadata.hpp"
#include "reflect/registry.hpp"
#include "reflect/container/object_proxy"
#include "reflect/container/arraylist"
#include <zeno/utils/helper.h>


namespace zeno {

    bool GlobalVariableStack::updateVariable(const GVariable& newvar)
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

    bool GlobalVariableStack::overrideVariable(const GVariable& var)
    {
        if (var.name.empty()) {
            return false;
        }
        auto it = GlobalVariables.find(var.name);
        if (it == GlobalVariables.end()) {
            GlobalVariables.insert(std::make_pair(var.name, OverrdeVector(std::stack<GVariable>({var}), var.gvar.type()) ));
            return true;
        }
        else {
            if (it->second.variableType.equal_fast(var.gvar.type())) {
                it->second.stack.push(var);
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
            if (!it->second.stack.empty()) {
                oldlVar = it->second.stack.top();
                it->second.stack.pop();
            }
        } 
    }

    zeno::reflect::Any GlobalVariableStack::getVariable(std::string varname)
    {
        auto it = GlobalVariables.find(varname);
        if (it != GlobalVariables.end()) {
            if (!it->second.stack.empty()) {
                return it->second.stack.top().gvar;
            }
        }
        return zeno::reflect::Any();
    }

    ZENO_API bool GlobalVariableManager::updateVariable(const GVariable& newvar)
    {
        return globalVariableStack.updateVariable(newvar);
    }

    ZENO_API bool GlobalVariableManager::overrideVariable(const GVariable& var)
    {
        return globalVariableStack.overrideVariable(var);
    }

    void GlobalVariableManager::cancelOverride(std::string varname, GVariable& cancelVar)
    {
        globalVariableStack.cancelOverride(varname, cancelVar);
    }

    ZENO_API zeno::reflect::Any GlobalVariableManager::getVariable(std::string varname)
    {
        return globalVariableStack.getVariable(varname);
    }

    ZENO_API GlobalVariableOverride::GlobalVariableOverride(std::weak_ptr<INode> wknode, std::string gvarName, zeno::reflect::Any var): currNode(wknode)
    {
        gvar = GVariable(gvarName, var);
        overrideSuccess = zeno::getSession().globalVariableManager->overrideVariable(gvar);
        if (overrideSuccess) {
            propagateDirty(currNode.lock(), gvar.name);
        }
    }

    ZENO_API GlobalVariableOverride::~GlobalVariableOverride()
    {
        if (overrideSuccess) {
            GVariable oldvar;
            zeno::getSession().globalVariableManager->cancelOverride(gvar.name, oldvar);
        }
    }

    ZENO_API bool GlobalVariableOverride::updateGlobalVariable(GVariable globalVariable)
    {
        if (zeno::getSession().globalVariableManager->updateVariable(globalVariable)) {
            propagateDirty(currNode.lock(), gvar.name);
            return true;
        }
        return false;
    }

}
