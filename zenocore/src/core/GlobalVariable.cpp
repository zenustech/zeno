#include <zeno/core/GlobalVariable.h>
#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>

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

    void GlobalVariableManager::propagateDirty(INode* spCurrNode, GVariable globalvar)
    {
        std::set<ObjPath> depNodes;
        std::set<ObjPath> upstreams;
        {
            getUpstreamNodes(spCurrNode, depNodes, upstreams);
            for (auto& objPath : depNodes) {
                auto it = globalVariablesNameTypeMap.find(objPath);
                if (it != globalVariablesNameTypeMap.end())
                {
                    for (auto& [name, type] : it->second) {
                        if (name == globalvar.name && type.equal_fast(globalvar.gvar.type())) {
                            if (auto node = zeno::getSession().mainGraph->getNodeByUuidPath(objPath)) {
                                auto pNode = AnyToINodePtr(node);
                                mark_dirty_by_dependNodes(pNode, true, upstreams);
                            }
                        }
                    }
                }
            }
        }
    }

    void GlobalVariableManager::getUpstreamNodes(INode* spCurrNode, std::set<ObjPath>& depNodes, std::set<ObjPath>& upstreams, std::string outParamName)
    {
        auto it = globalVariablesNameTypeMap.find(spCurrNode->get_uuid_path());
        if (it != globalVariablesNameTypeMap.end() && !it->second.empty()) {
            depNodes.insert(spCurrNode->get_uuid_path());
        }

        if (upstreams.find(spCurrNode->get_uuid_path()) != upstreams.end()) {
            return;
        }
        if (SubnetNode* pSubnetNode = dynamic_cast<SubnetNode*>(spCurrNode))
        {
            auto suboutoutGetUpstreamFunc = [&pSubnetNode, &depNodes, &upstreams, this](std::string paramName) {
                if (auto _suboutput = pSubnetNode->subgraph->getNode(paramName)) {
                    auto suboutput = AnyToINodePtr(_suboutput);
                    getUpstreamNodes(suboutput, depNodes, upstreams);
                    upstreams.insert(suboutput->get_uuid_path());
                }
            };
            if (outParamName != "") {
                suboutoutGetUpstreamFunc(outParamName);
            }
            else {
                for (auto& param : pSubnetNode->get_output_primitive_params()) {
                    suboutoutGetUpstreamFunc(param.name);
                }
                for (auto& param : pSubnetNode->get_output_object_params()) {
                    suboutoutGetUpstreamFunc(param.name);
                }
            }
            upstreams.insert(pSubnetNode->get_uuid_path());
        }
        else {
            auto spGraph = spCurrNode->getGraph().lock();
            for (auto& param : spCurrNode->get_input_primitive_params()) {
                for (auto link : param.links) {
                    if (spGraph)
                    {
                        auto outParam = link.outParam;
                        auto outNode = AnyToINodePtr(spGraph->getNode(link.outNode));
                        assert(outNode);
                        getUpstreamNodes(outNode, depNodes, upstreams, outParam);
                        upstreams.insert(outNode->get_uuid_path());
                    }
                }
            }
            for (auto& param : spCurrNode->get_input_object_params()) {
                for (auto link : param.links) {
                    if (spGraph)
                    {
                        auto outParam = link.outParam;
                        auto outNode = AnyToINodePtr(spGraph->getNode(link.outNode));
                        assert(outNode);
                        getUpstreamNodes(outNode, depNodes, upstreams, outParam);
                        upstreams.insert(outNode->get_uuid_path());
                    }
                }
            }
            upstreams.insert(spCurrNode->get_uuid_path());
        }
        std::shared_ptr<Graph> spGraph = spCurrNode->getGraph().lock();
        assert(spGraph);
        auto parentSubgNode = spGraph->getParentSubnetNode();
        if (parentSubgNode && spCurrNode->get_nodecls() == "SubInput")
        {
            upstreams.insert(parentSubgNode->get_uuid_path());
            auto parentSubgNodeGetUpstreamFunc = [&depNodes, &upstreams, &parentSubgNode, this](std::string outParam) {
                auto outNode = AnyToINodePtr(parentSubgNode->get_graph()->getNode(outParam));
                assert(outNode);
                getUpstreamNodes(outNode, depNodes, upstreams, outParam);
                upstreams.insert(outNode->get_uuid_path());
            };
            bool find = false;
            const auto& parentSubgNodePrimsInput = parentSubgNode->get_input_prim_param(spCurrNode->get_name(), &find);
            if (find) {
                for (auto link : parentSubgNodePrimsInput.links) {
                    parentSubgNodeGetUpstreamFunc(link.outParam);
                }
            }
            bool find2 = false;
            const auto& parentSubgNodeObjsInput = parentSubgNode->get_input_obj_param(spCurrNode->get_name(), &find2);
            if (find) {
                for (auto link : parentSubgNodeObjsInput.links) {
                    parentSubgNodeGetUpstreamFunc(link.outParam);
                }
            }
        }
    }

    void GlobalVariableManager::mark_dirty_by_dependNodes(INode* spCurrNode, bool bOn, std::set<ObjPath> nodesRange, std::string inParamName)
    {
        if (!nodesRange.empty()) {
            if (nodesRange.find(spCurrNode->get_uuid_path()) == nodesRange.end()) {
                return;
            }
        }
        
        if (spCurrNode->is_dirty())
            return;
        spCurrNode->mark_dirty(true, true, false);

        if (bOn) {
            auto spGraph = spCurrNode->getGraph().lock();
            for (auto& param : spCurrNode->get_output_primitive_params()) {
                for (auto link : param.links) {
                    if (spGraph) {
                        auto inParam = link.inParam;
                        auto inNode = AnyToINodePtr(spGraph->getNode(link.inNode));
                        assert(inNode);
                        mark_dirty_by_dependNodes(inNode, bOn, nodesRange, inParam);
                    }
                }
            }
            for (auto& param : spCurrNode->get_output_object_params()) {
                for (auto link : param.links) {
                    if (spGraph) {
                        auto inParam = link.inParam;
                        auto inNode = AnyToINodePtr(spGraph->getNode(link.inNode));
                        assert(inNode);
                        mark_dirty_by_dependNodes(inNode, bOn, nodesRange, inParam);
                    }
                }
            }
        }

        if (auto pSubnetNode = dynamic_cast<SubnetNode*>(spCurrNode))
        {
            auto subinputMarkDirty = [&pSubnetNode, &nodesRange, this](bool dirty, std::string paramName) {
                if (auto subinput = pSubnetNode->subgraph->getNode(paramName))
                {
                    auto pNode = AnyToINodePtr(subinput);
                    mark_dirty_by_dependNodes(pNode, dirty, nodesRange);
                }
            };
            if (inParamName != "") {
                subinputMarkDirty(bOn, inParamName);
            }
            else {
                for (auto& param : pSubnetNode->get_input_primitive_params())
                    subinputMarkDirty(bOn, param.name);
                for (auto& param : pSubnetNode->get_input_object_params())
                    subinputMarkDirty(bOn, param.name);
            }
        }

        std::shared_ptr<Graph> spGraph = spCurrNode->getGraph().lock();
        assert(spGraph);
        auto parentSubgNode = spGraph->getParentSubnetNode();
        if (parentSubgNode && spCurrNode->get_nodecls() == "SubOutput")
        {
            auto parentSubgNodeMarkDirty = [&nodesRange, &parentSubgNode, this](std::string inParam) {
                auto inNode = AnyToINodePtr(parentSubgNode->get_graph()->getNode(inParam));
                mark_dirty_by_dependNodes(inNode, true, nodesRange, inParam);
            };
            bool find = false;
            const auto& parentSubgNodeOutputPrim = parentSubgNode->get_output_prim_param(spCurrNode->get_name(), &find);
            if (find) {
                for (auto link : parentSubgNodeOutputPrim.links) {
                    parentSubgNodeMarkDirty(link.inParam);
                }
            }
            bool find2 = false;
            const auto& parentSubgNodeOutputObjs = parentSubgNode->get_output_obj_param(spCurrNode->get_name(), &find);
            if (find2) {
                for (auto link : parentSubgNodeOutputObjs.links) {
                    parentSubgNodeMarkDirty(link.inParam);
                }
            }
            parentSubgNode->mark_dirty(true, true, false);
        }
    }

    void GlobalVariableManager::removeDependGlobalVaraible(const ObjPath& nodepath, std::string name)
    {
        auto it = globalVariablesNameTypeMap.find(nodepath);
        if (it != globalVariablesNameTypeMap.end())
        {
            it->second.erase(name);
        }
    }

    void GlobalVariableManager::addDependGlobalVaraible(const ObjPath& nodepath, std::string name, zeno::reflect::RTTITypeInfo type)
    {
        auto it = globalVariablesNameTypeMap.find(nodepath);
        if (it == globalVariablesNameTypeMap.end())
        {
            globalVariablesNameTypeMap.insert({ nodepath, std::map<std::string, zeno::reflect::RTTITypeInfo>{ {name, type}} });
        }
        else {
            it->second.insert({name, type});
        }
    }

    bool GlobalVariableManager::updateVariable(const GVariable& newvar)
    {
        return globalVariableStack.updateVariable(newvar);
    }

    bool GlobalVariableManager::overrideVariable(const GVariable& var)
    {
        return globalVariableStack.overrideVariable(var);
    }

    void GlobalVariableManager::cancelOverride(std::string varname, GVariable& cancelVar)
    {
        globalVariableStack.cancelOverride(varname, cancelVar);
    }

    zeno::reflect::Any GlobalVariableManager::getVariable(std::string varname)
    {
        return globalVariableStack.getVariable(varname);
    }

}