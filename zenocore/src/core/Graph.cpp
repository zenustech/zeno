#include <zeno/core/Graph.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INodeClass.h>
#include <zeno/core/Session.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/Assets.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GraphException.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/extra/GlobalError.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/extra/CalcContext.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>
#include <zeno/utils/uuid.h>
#include <zeno/utils/helper.h>
#include <iostream>
#include <regex>
#include <cctype>
#include <zeno/core/GlobalVariable.h>
#include <zeno/core/typeinfo.h>
#include <future>
#include "zeno_types/reflect/reflection.generated.hpp"
//#include <Python.h>
//#include <pybind11/pybind11.h>


namespace zeno {

Context::Context() = default;
Context::~Context() = default;

Context::Context(Context const &other)
    : visited(other.visited)
{}

Graph::Graph(const std::string& name, bool bAssets) : m_name(name), m_bAssets(bAssets) {
    
}

Graph::~Graph() {

}

void Graph::clearNodes() {
    m_nodes.clear();
}

void Graph::clearContainerUpdateInfo() {
    for (const auto& [uuid, node] : m_nodes) {
        node->clear_container_info();
    }
}

void Graph::addNode(std::string const &cls, std::string const &id) {
    //todo: deprecated.
#if 0
    if (nodes.find(id) != nodes.end())
        return;  // no add twice, to prevent output object invalid
    auto cl = safe_at(session->nodeClasses, cls, "node class name").get();
    auto node = cl->new_instance(id);
    node->graph = this;
    node->name = id;
    node->nodeClass = cl;
    nodes[id] = std::move(node);
#endif
}

Graph *Graph::getSubnetGraph(std::string const & node_name) const {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    NodeImpl* pNode = safe_at(m_nodes, uuid, "node name").get();
    auto node = dynamic_cast<SubnetNode*>(pNode);
    return node ? node->get_subgraph() : nullptr;
}

render_update_info Graph::applyNode(std::string const &node_name, CalcContext* pContext) {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    auto node = safe_at(m_nodes, uuid, "node name").get();
    if (!node->is_dirty()) {
        return render_update_info();
    }

    node->execute(pContext);

    render_update_info info;
    if (node->is_view()) {
        info.reason = Update_Reconstruct;
        auto pObj = node->get_default_output_object();
        if (pObj) {
            info.spObject = pObj->clone();
        }
        info.uuidpath_node_objkey = node->get_uuid_path();

        if (node->is_nocache()) {
            //销毁对象，所属权已经移交到info上
            node->mark_takeover();
        }
    }
    return info;
}

void Graph::mark_clean() {
    for (auto& [_, node] : m_nodes) {
        if (node->get_run_status() == Node_RunSucceed) {
            node->mark_clean();
        }
    }
}

void Graph::applyNodes(std::set<std::string> const &nodes, render_reload_info& infos) {
    auto launch_method = zeno::getSession().is_async_executing() ? 
        (std::launch::async | std::launch::deferred) : std::launch::deferred;
    clearContainerUpdateInfo();

    CalcContext ctx;
    std::vector<std::future<render_update_info>> tasks;
    for (auto const& node_name: nodes) {
        tasks.push_back(std::async(launch_method, &Graph::applyNode, this, node_name, &ctx));
    }
    for (auto& task : tasks) {
        if (task.valid()) {
            task.wait();
        }
    }

    for (auto& task : tasks) {
        const zeno::render_update_info& info = task.get();
        if (!info.uuidpath_node_objkey.empty()) {
            infos.objs.push_back(info);
        }
    }
    infos.policy = Reload_Calculation;
}

void Graph::runGraph(render_reload_info& infos) {
    log_debug("{} nodes to exec", m_viewnodes.size());
    return applyNodes(m_viewnodes, infos);
}

void Graph::onNodeParamUpdated(PrimitiveParam* spParam, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
    auto spNode = spParam->m_wpNode;
    assert(spNode);
    const std::string& uuid = spNode->get_uuid();
    const std::string& cls = spNode->get_nodecls();
    bool bHasFrameRel = spNode->has_frame_relative_params();
    if (bHasFrameRel) {
        frame_nodes.insert(uuid);
    }
    else {
        if (!zeno::getSession().is_frame_node(cls)) {
            frame_nodes.erase(uuid);
        }
    }
}

void Graph::parseNodeParamDependency(PrimitiveParam* spParam, zeno::reflect::Any& new_value)
{
    auto spNode = spParam->m_wpNode;
    assert(spNode);
    if (!spParam->defl.has_value()) {
        return;
    }
    const std::string& uuid = spNode->get_uuid();
    if (gParamType_String == spParam->type)
    {
        std::string defl = zeno::any_cast_to_string(spParam->defl);
        std::regex pattern("\\$F");
        if (std::regex_search(defl, pattern, std::regex_constants::match_default)) {
            frame_nodes.insert(uuid);
        }
    }
    else if (gParamType_Int == spParam->type || gParamType_Float == spParam->type)
    {
        assert(gParamType_PrimVariant == spParam->defl.type().hash_code());
        const zeno::PrimVar& editVar = zeno::reflect::any_cast<zeno::PrimVar>(spParam->defl);
        std::visit([=](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                std::regex pattern("\\$F");
                if (std::regex_search(arg, pattern, std::regex_constants::match_default)) {
                    frame_nodes.insert(uuid);
                }
            }
        }, editVar);
    }
    else if (gParamType_Vec2f == spParam->type ||
        gParamType_Vec2i == spParam->type ||
        gParamType_Vec3f == spParam->type ||
        gParamType_Vec3i == spParam->type ||
        gParamType_Vec4f == spParam->type ||
        gParamType_Vec4i == spParam->type)
    {
        assert(gParamType_VecEdit == spParam->defl.type().hash_code());
        const zeno::vecvar& editVar = zeno::reflect::any_cast<zeno::vecvar>(spParam->defl);
        for (auto primvar : editVar) {
            std::visit([=](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    std::regex pattern("\\$F");
                    if (std::regex_search(arg, pattern, std::regex_constants::match_default)) {
                        frame_nodes.insert(uuid);
                    }
                }
            }, primvar);
        }
    }
}

bool Graph::isFrameNode(std::string uuid)
{
    return frame_nodes.count(uuid);
}

void Graph::viewNodeUpdated(const std::string node, bool bView) {
    if (bView) {
        //TODO: only run calculation chain which associate with `node`.
        //getSession().run_main_graph();
        //disable the previous view.
#if 0
        auto viewnodes = m_viewnodes;
        for (auto nodename : viewnodes) {
            auto spNode = getNode(nodename);
            spNode->set_view(false);
        }
#endif
        m_viewnodes.insert(node);
    }
    else {
        m_viewnodes.erase(node);
        //TODO: update objsmanager to hide objs.
    }
}

void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    //safe_at(nodes, dn, "node name")->inputBounds[ds] = std::pair(sn, ss);
}

void Graph::setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    //todo: deprecated.
    //safe_at(nodes, id, "node name")->inputs[par] = val;
}

void Graph::setKeyFrame(std::string const &id, std::string const &par, zany const &val) {
    //todo: deprecated.
    /*
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->kframes.insert(par);
    */
}

void Graph::setFormula(std::string const &id, std::string const &par, zany const &val) {
    //todo: deprecated.
    /*
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->formulas.insert(par);
    */
}


std::map<std::string, zany> Graph::callSubnetNode(std::string const &id,
        std::map<std::string, zany> inputs) const {
    //todo: deprecated.
    return std::map<std::string, zany>();
}

void Graph::addNodeOutput(std::string const& id, std::string const& par) {
    // add "dynamic" output which is not descriped by core.
    //todo: deprecated.
    //safe_at(nodes, id, "node name")->outputs[par] = nullptr;
}

static void initSpecialNode(zeno::NodeImpl* pNodeImpl, const NodeData& node) {
    if (node.cls == "FlipSolver") {
        //节点在初始化的时候是脏的，但还需要手动触发solver的dirty_changed，让它删cache
        pNodeImpl->dirty_changed(true, Dirty_All, false, false);
    }
        if (node.cls == "SubInput") {
            //TODO
        }
        else if (node.cls == "SubOutput") {
            //TODO
        }
        else if (node.cls == "Group") {
            if (node.group.has_value()) {
            pNodeImpl->update_param("title", node.group->title);
            pNodeImpl->update_param("background", node.group->background);
            pNodeImpl->update_param("size", node.group->sz);
            pNodeImpl->update_param("items", join_str(node.group->items, ","));
            }
        }
        else if (zeno::isDerivedFromSubnetNodeName(node.cls))
        {
        if (auto sbn = dynamic_cast<SubnetNode*>(pNodeImpl))
                sbn->setCustomUi(node.customUi);
        }
    }

void Graph::init(const GraphData& graph) {
    auto& sess = getSession();
    sess.setApiLevelEnable(false);
    zeno::scope_exit([&]() {
        sess.setApiLevelEnable(true);
    });

    m_name = graph.name;
    //import nodes first.
    for (const auto& [name, node] : graph.nodes) {
        bool bAssets = node.asset.has_value();
        bool bAssetLock = node.bLocked;
        auto spNode = createNode(node.cls, name, bAssets, node.uipos, true, bAssets ? &bAssetLock : nullptr);
        spNode->init(node);
        initSpecialNode(spNode, node);
        sess.reportIOProgress(name, 1);
    }
    //import edges
    for (const auto& link : graph.links) {
        if (!isLinkValid(link))
            continue;
        auto outNode = getNode(link.outNode);
        auto inNode = getNode(link.inNode);

        bool bExist = false;
        bool bOutputPrim = outNode->isPrimitiveType(false, link.outParam, bExist);
        bool bInputPrim = inNode->isPrimitiveType(true, link.inParam, bExist);

        if (bInputPrim) {
            std::shared_ptr<PrimitiveLink> spLink = std::make_shared<PrimitiveLink>();
            outNode->init_primitive_link(false, link.outParam, spLink, link.targetParam);
            inNode->init_primitive_link(true, link.inParam, spLink, link.targetParam);
        }
        else {
            std::shared_ptr<ObjectLink> spLink = std::make_shared<ObjectLink>();
            spLink->fromkey = link.outKey;
            spLink->tokey = link.inKey;
            outNode->init_object_link(false, link.outParam, spLink, link.targetParam);
            inNode->init_object_link(true, link.inParam, spLink, link.targetParam);
        }
    }
}

void Graph::initRef(const GraphData& graph) {
    for (const auto& [nodename, refparams] : graph.references) {
        auto refNode = getNode(nodename);
        const auto& uuidpath = refNode->get_uuid_path();
        for (auto paramname : refparams) {
            refNode->constructReference(paramname);
        }
    }
    for (const std::string& subnetnode : subnet_nodes) {
        auto pNodeImpl = m_nodes[subnetnode].get();
        auto spSubnetNode = dynamic_cast<SubnetNode*>(pNodeImpl);
        assert(spSubnetNode);
        if (spSubnetNode) {
            NodesData nodes = graph.nodes;
            const std::string& nodename = pNodeImpl->get_name();
            const NodeData& node = nodes[nodename];
            const std::optional<GraphData>& optSubg = node.subgraph;
            if (optSubg.has_value()) {
                const GraphData& subg = optSubg.value();
                spSubnetNode->get_subgraph()->initRef(subg);
            }
        }
    }
}

void Graph::markDirtyWhenFrameChanged()
{
    for (const std::string& uuid : frame_nodes) {
        auto pNode = m_nodes[uuid].get();
        assert(pNode);
        auto pNodeImpl = pNode;
        pNode->mark_dirty(true, Dirty_FrameChanged);
        }
    std::set<std::string> nodes = subnet_nodes;
    nodes.insert(asset_nodes.begin(), asset_nodes.end());
    for (const std::string& uuid : nodes) {
        auto pNode = m_nodes[uuid].get();
        assert(pNode);
        auto pNodeImpl = pNode;
        auto spSubnetNode = dynamic_cast<SubnetNode*>(pNodeImpl);
        spSubnetNode->get_subgraph()->markDirtyWhenFrameChanged();
    }
}

void Graph::markDirtyAndCleanup()
{
    for (const auto& [uuid, node] : m_nodes) {
        node->mark_dirty(true);
        node->clearCalcResults();  //clear all result prim and objs
    }
}

std::string Graph::generateNewName(const std::string& node_cls, const std::string& origin_name, bool bAssets)
{
    if (node_set.find(node_cls) == node_set.end())
        node_set.insert(std::make_pair(node_cls, std::set<std::string>()));

    auto& nodes = node_set[node_cls];

    if (!origin_name.empty() && m_name2uuid.find(origin_name) == m_name2uuid.end())
    {
        nodes.insert(origin_name);
        return origin_name;
    }

    std::string tempName = node_cls;
    if (!bAssets) {
        auto& nodeClass = getSession().nodeClasses;
        auto it = nodeClass.find(node_cls);
        if (it != nodeClass.end()) {
            auto cl = it->second.get();
            if (cl && !cl->m_customui.nickname.empty())
                tempName = cl->m_customui.nickname;
        }
    }

    int i = 1;
    bool end_with_digit = std::isdigit(tempName.back());
    while (true) {
        std::string new_name = tempName + (end_with_digit ? "_" : "") + std::to_string(i++);
        if (nodes.find(new_name) == nodes.end()) {
            nodes.insert(new_name);
            return new_name;
        }
    }
    return "";
}

void Graph::updateWildCardParamTypeRecursive(Graph* spCurrGarph, NodeImpl* spNode, std::string paramName, bool bPrim, bool bInput, ParamType newtype)
{
    if (!spCurrGarph || !spNode)
        return;
    if (spNode->get_nodecls() == "SubOutput" || spNode->get_nodecls() == "SubInput") {
        spNode->update_param_type(paramName, bPrim, bInput, newtype);
        auto links = spNode->getLinksByParam(bInput, paramName);
        for (auto& link : links) {
            if (bInput) {}
                //updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.outNode), link.outParam, bPrim, !bInput, newtype);
            else {
                if (auto innode = spCurrGarph->getNode(link.inNode)) {
                    ParamType paramType;
                    SocketType socketType;
                    bool bWildcard = false;
                    innode->getParamTypeAndSocketType(link.inParam, bPrim, !bInput, paramType, socketType, bWildcard);
                    if (bWildcard || (!bPrim && isSubnetInputOutputParam(innode, link.inParam)))
                        updateWildCardParamTypeRecursive(spCurrGarph, innode, link.inParam, bPrim, !bInput, newtype);
                    else if (paramType != newtype)
                        spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                }
            }
        }
        if (auto graph = spNode->getGraph()) {
            if (NodeImpl* pNodeImpl = getParentSubnetNode()) {
                if (SubnetNode* parentSubgNode = dynamic_cast<SubnetNode*>(pNodeImpl)) {
                    pNodeImpl->update_param_type(spNode->get_name(), bPrim, !bInput, newtype);
                    for (auto& link : pNodeImpl->getLinksByParam(!bInput, spNode->get_name())) {
                        if (auto parentGraph = pNodeImpl->getGraph()) {
                            auto const& inNode = parentGraph->getNode(link.inNode);
                            auto const& outNode = parentGraph->getNode(link.outNode);
                            ParamType inNodeParamType;
                            SocketType inNodeSocketType;
                            ParamType outNodeParamType;
                            SocketType outNodeSocketType;
                            bool bInWildcard = false, bOutWildcard = false;
                            inNode->getParamTypeAndSocketType(link.inParam, bPrim, true, inNodeParamType, inNodeSocketType, bInWildcard);
                            outNode->getParamTypeAndSocketType(link.outParam, bPrim, false, outNodeParamType, outNodeSocketType, bOutWildcard);
                            if (inNodeParamType != outNodeParamType) {
                                if (bInWildcard && !(!bPrim && isSubnetInputOutputParam(inNode, link.inParam)))
                                    parentGraph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                                else {
                                    if (bInput)
                                        updateWildCardParamTypeRecursive(parentGraph, inNode, link.inParam, bPrim, bInput, newtype);
                                    else {
                                        if (bOutWildcard || (!bPrim && isSubnetInputOutputParam(outNode, link.outParam)))
                                            updateWildCardParamTypeRecursive(parentGraph, outNode, link.outParam, bPrim, bInput, newtype);
                                        else
                                            parentGraph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (SubnetNode* subnet = dynamic_cast<SubnetNode*>(spNode)) {
        spNode->update_param_type(paramName, bPrim, bInput, newtype);
            for (auto& link : spNode->getLinksByParam(bInput, paramName)) {
                if (bInput) {}
                //updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.outNode), link.outParam, bPrim, !bInput, newtype);
                else {
                    if (auto innode = spCurrGarph->getNode(link.inNode)) {
                        ParamType paramType;
                        SocketType socketType;
                        bool bWildcard = false;
                        innode->getParamTypeAndSocketType(link.inParam, bPrim, !bInput, paramType, socketType, bWildcard);
                        if (bWildcard || (!bPrim && isSubnetInputOutputParam(innode, link.inParam)))
                            updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.inNode), link.inParam, bPrim, !bInput, newtype);
                        else if (paramType != newtype)
                            spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                    }
                }
            }
            if (auto innerNode = subnet->get_subgraph()->getNode(paramName)) {
                std::vector<std::string> inparamNames;
                if (bInput) {
                    for (auto& param : innerNode->get_output_object_params())
                        inparamNames.emplace_back(param.name);
                }
                else {
                    if (bPrim)
                    for (auto& param : innerNode->get_input_primitive_params())
                            inparamNames.emplace_back(param.name);
                    else
                        for (auto& param : innerNode->get_input_object_params())
                            inparamNames.emplace_back(param.name);
                }
            for (auto& name : inparamNames) {
                    innerNode->update_param_type(name, bPrim, !bInput, newtype);
                for (auto& link : innerNode->getLinksByParam(!bInput, name)) {
                        auto const& inNode = subnet->get_subgraph()->getNode(link.inNode);
                        auto const& outNode = subnet->get_subgraph()->getNode(link.outNode);
                        ParamType inNodeParamType;
                        SocketType inNodeSocketType;
                        ParamType outNodeParamType;
                        SocketType outNodeSocketType;
                        bool bInWildcard = false, bOutWildcard = false;
                        inNode->getParamTypeAndSocketType(link.inParam, bPrim, true, inNodeParamType, inNodeSocketType, bInWildcard);
                        outNode->getParamTypeAndSocketType(link.outParam, bPrim, false, outNodeParamType, outNodeSocketType, bOutWildcard);
                        if (inNodeParamType != outNodeParamType) {
                            if (bInWildcard && !(!bPrim && isSubnetInputOutputParam(inNode, link.inParam)))
                                subnet->get_subgraph()->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                            else {
                                if (bInput)
                                    updateWildCardParamTypeRecursive(subnet->get_subgraph(), inNode, link.inParam, bPrim, bInput, newtype);
                                else {
                                    if (bOutWildcard || (!bPrim && isSubnetInputOutputParam(outNode, link.outParam)))
                                        updateWildCardParamTypeRecursive(subnet->get_subgraph(), outNode, link.outParam, bPrim, bInput, newtype);
                                    else
                                        subnet->get_subgraph()->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                                }
                            }
                        }
                    }
                }
            }
        }
    else {
        const auto& params = spNode->getWildCardParams(paramName, bPrim);
        for (const auto& param : params) {
            spNode->update_param_type(param.first, bPrim, param.second, newtype);
            for (auto& link : spNode->getLinksByParam(param.second, param.first)) {
                NodeImpl* otherNodeLinkToThis = param.second ? spCurrGarph->getNode(link.outNode) : spCurrGarph->getNode(link.inNode);
                if (otherNodeLinkToThis) {
                    if (param.second) {
                        ParamType paramType;
                        SocketType socketType;
                        bool bWildcard = false;
                        otherNodeLinkToThis->getParamTypeAndSocketType(link.outParam, bPrim, false, paramType, socketType, bWildcard);
                        if (paramType != newtype) {
                            if (bWildcard || (!bPrim && isSubnetInputOutputParam(otherNodeLinkToThis, link.outParam)))
                                updateWildCardParamTypeRecursive(spCurrGarph, otherNodeLinkToThis, link.outParam, bPrim, false, newtype);
                            else
                                spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                        }
                    }
                    else {
                        ParamType paramType;
                        SocketType socketType;
                        bool bWildcard = false;
                        otherNodeLinkToThis->getParamTypeAndSocketType(link.inParam, bPrim, true, paramType, socketType, bWildcard);
                        if (paramType != newtype) {
                            bool bWildcard = otherNodeLinkToThis->get_input_param(link.inParam).bWildcard;
                            if (bWildcard || (!bPrim && isSubnetInputOutputParam(otherNodeLinkToThis, link.inParam)))
                                updateWildCardParamTypeRecursive(spCurrGarph, otherNodeLinkToThis, link.inParam, bPrim, true, newtype);
                            else
                                spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                        }
                    }
                }
            }
        }
    }
}

void Graph::removeLinkWhenUpdateWildCardParam(const std::string& outNodeName, const std::string& inNodeName, EdgeInfo& edge)
{
    NodeImpl* outNode = getNode(outNodeName);
    NodeImpl* inNode = getNode(inNodeName);
    if (!outNode || !inNode)
        return;
    outNode->removeLink(false, edge);
    inNode->removeLink(true, edge);
    inNode->mark_dirty(true);
    CALLBACK_NOTIFY(removeLink, edge)
}

void Graph::resetWildCardParamsType(bool bParamWildcard, NodeImpl* node, const std::string& paramName, const bool& bPrimType, const bool& bInput)
{
    if (!node)
        return;
    std::function<bool(Graph*, NodeImpl*, std::string, bool, std::set<std::string>&)> linkedToSpecificType =
        [&linkedToSpecificType, this](Graph* currGraph, NodeImpl* node, std::string paramName, bool bPrimType, std::set<std::string>& visited)->bool {
        const auto& params = node->getWildCardParams(paramName, bPrimType);
        for (auto& param : params) {
            visited.insert(node->get_uuid() + param.first);

            if (node->get_nodecls() == "SubOutput" || node->get_nodecls() == "SubInput") {
                auto links = node->getLinksByParam(param.second, paramName);
                for (auto& link : links) {
                    if (param.second) {
                        const auto& outNode = currGraph->getNode(link.outNode);
                        if (!visited.count(outNode->get_uuid() + link.outParam))
                            if (linkedToSpecificType(currGraph, outNode, link.outParam, bPrimType, visited))
                                return true;
                    }
                    else {
                        const auto& inNode = currGraph->getNode(link.inNode);
                        if (!visited.count(inNode->get_uuid() + link.inParam))
                            if (linkedToSpecificType(currGraph, inNode, link.inParam, bPrimType, visited))
                                return true;
                    }
                }
                if (auto graph = node->getGraph()) {
                    if (NodeImpl* pNodeImpl = getParentSubnetNode()) {
                        SubnetNode* parentSubgNode = getSubnetNode(getParentSubnetNode());
                        if (parentSubgNode) {
                            visited.insert(pNodeImpl->get_uuid() + node->get_uuid());
                            if (auto parentGraph = pNodeImpl->getGraph()) {
                                for (auto& link : pNodeImpl->getLinksByParam(!param.second, node->get_name())) {
                                    const auto& node = parentGraph->getNode(param.second ? link.inNode : link.outNode);
                                    ParamType paramType;
                                    SocketType socketType;
                                    bool bWildcard = false;
                                    node->getParamTypeAndSocketType(param.second ? link.inParam : link.outParam, bPrimType, param.second, paramType, socketType, bWildcard);
                                    if (bWildcard && !(!bPrimType && isSubnetInputOutputParam(node, param.second ? link.inParam : link.outParam)))
                                        return true;
                                    else {
                                        if (!visited.count(node->get_uuid() + (param.second ? link.inParam : link.outParam))) {
                                            if (linkedToSpecificType(parentGraph, node, param.second ? link.inParam : link.outParam, bPrimType, visited))
                                                return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (zeno::isDerivedFromSubnetNodeName(node->get_nodecls())) {
                if (SubnetNode* subnet = dynamic_cast<SubnetNode*>(node)) {
                    if (auto innerNode = subnet->get_subgraph()->getNode(paramName)) {
                        std::vector<std::string> inparamNames;
                        if (param.second) {
                            for (auto& param : innerNode->get_output_object_params())
                                inparamNames.emplace_back(param.name);
                        }
                        else {
                            if (bPrimType)
                                for (auto& param : innerNode->get_input_primitive_params())
                                    inparamNames.emplace_back(param.name);
                            else
                                for (auto& param : innerNode->get_input_object_params())
                                    inparamNames.emplace_back(param.name);
                        }
                        for (auto& name : inparamNames) {
                            for (auto& link : innerNode->getLinksByParam(!param.second, name)) {
                                auto node = subnet->get_subgraph()->getNode(param.second ? link.inNode : link.outNode);
                                ParamType paramType;
                                SocketType socketType;
                                bool bWildcard = false;
                                node->getParamTypeAndSocketType(param.second ? link.inParam : link.outParam, bPrimType, param.second, paramType, socketType, bWildcard);
                                if (bWildcard && !(!bPrimType && isSubnetInputOutputParam(node, param.second ? link.inParam : link.outParam)))
                                    return true;
                                else {
                                    if (!visited.count(node->get_uuid() + (param.second ? link.inParam : link.outParam)))
                                        if (linkedToSpecificType(subnet->get_subgraph(), node, param.second ? link.inParam : link.outParam, bPrimType, visited))
                                            return true;
                                }
                            }
                        }
                    }
                }
            }
            else {
                for (auto& link : node->getLinksByParam(param.second, param.first)) {
                    const auto& node = currGraph->getNode(param.second ? link.outNode : link.inNode);
                    ParamType paramType;
                    SocketType socketType;
                    bool bWildcard = false;
                    node->getParamTypeAndSocketType(param.second ? link.outParam : link.inParam, bPrimType, !param.second, paramType, socketType, bWildcard);
                    if (bWildcard && !(!bPrimType && isSubnetInputOutputParam(node, param.second ? link.inParam : link.outParam)))
                        return true;
                    else {
                        if (!visited.count(node->get_uuid() + (param.second ? link.outParam : link.inParam)))
                            if (linkedToSpecificType(currGraph, node, param.second ? link.outParam : link.inParam, bPrimType, visited))
                                return true;
                    }
                }

            }
        }
        return false;
    };
    if (bParamWildcard || (!bPrimType && isSubnetInputOutputParam(node, paramName))) {
        std::set<std::string> visited;
        if (linkedToSpecificType(this, node, paramName, bPrimType, visited))
            return;
    }
}

bool Graph::isAssets() const
{
    return m_bAssets;
}

bool Graph::isAssetRoot() const
{
    NodeImpl* pNode = m_parSubnetNode;
    const Graph* pGraph = this;
    while (pNode) {
        pGraph = pNode->getGraph();
        pNode = pGraph->m_parSubnetNode;
    }
    return pGraph->m_bAssets;
}

std::set<std::string> Graph::searchByClass(const std::string& name) const
{
    auto it = node_set.find(name);
    if (it == node_set.end())
        return {};
    return it->second;
}

std::string Graph::updateNodeName(const std::string oldName, const std::string newName)
{
    if (oldName == newName)
        return "";

    CORE_API_BATCH

    auto sync_to_set = [=](std::set<std::string>& nodes, std::string oldName, std::string newName) {
        if (nodes.find(oldName) != nodes.end()) {
            nodes.erase(oldName);
            nodes.insert(newName);
        }
    };

    if (newName.empty())
        return "";

    const std::string uuid = safe_at(m_name2uuid, oldName, "uuid");
    if (m_nodes.find(uuid) == m_nodes.end()) {
        return "";
    }

    NodeImpl* spNode = m_nodes[uuid].get();
    std::string oldPath = spNode->get_path();
    std::string name = newName;
    if (m_name2uuid.find(name) != m_name2uuid.end()) {
        name = generateNewName(spNode->get_nodecls());
    }
    spNode->set_name(name);

    m_name2uuid[name] = m_name2uuid[oldName];
    m_name2uuid.erase(oldName);

    sync_to_set(m_viewnodes, oldName, name);

    spNode->onNodeNameUpdated(oldName, name);
    spNode->mark_dirty(true);

    CALLBACK_NOTIFY(updateNodeName, oldName, name)
    return name;
}

void Graph::clear()
{
    CALLBACK_NOTIFY(clear)

    m_nodes.clear();
    nodesToExec.clear();

    subInputNodes.clear();
    subOutputNodes.clear();
    m_name2uuid.clear();
    node_set.clear();
    frame_nodes.clear();
    subnet_nodes.clear();
    asset_nodes.clear();
    subinput_nodes.clear();
    suboutput_nodes.clear();
    m_viewnodes.clear();

    m_parSubnetNode = nullptr;
}

NodeImpl* Graph::createNode(
    const std::string& cls,
    const std::string& orgin_name,
    bool bAssets,
    std::pair<float, float> pos,
    bool isIOInit,
    bool* pbAssetLock
    )
{
    CORE_API_BATCH
    const std::string& name = generateNewName(cls, orgin_name, bAssets);

    if (name == "CopyAttribute1") {
        int j;
        j = 0;
    }

    std::string uuid;
    NodeImpl* pNode = nullptr;
    std::unique_ptr<NodeImpl> upNode;
    if (!bAssets) {
        auto& nodeClass = getSession().nodeClasses;
        std::string nodecls = cls;
        auto it = nodeClass.find(nodecls);
        if (it == nodeClass.end()) {
            upNode = std::make_unique<NodeImpl>(nullptr);   //空壳
            pNode = upNode.get();
            pNode->initUuid(this, nodecls);
            uuid = pNode->get_uuid();
        }
        else {
            INodeClass* cl = it->second.get();
            upNode = std::move(cl->new_instance(this, name));
            pNode = upNode.get();
            pNode->nodeClass = cl;
            uuid = pNode->get_uuid();
        }
    }
    else {
        bool isCurrentGraphAsset = getSession().assets->isAssetGraph(this);
        bool bAssetLocked = pbAssetLock ? *pbAssetLock : true;
        upNode = std::move(getSession().assets->newInstance(this, cls, name, isCurrentGraphAsset, bAssetLocked));
        pNode = upNode.get();
        uuid = pNode->get_uuid();
        asset_nodes.insert(uuid);
    }

    if (zeno::getSession().is_frame_node(cls)) {
        frame_nodes.insert(uuid);
    }
    if (zeno::isDerivedFromSubnetNodeName(cls)) {
        subnet_nodes.insert(uuid);
        if (!isIOInit) {
            zeno::ParamsUpdateInfo updateInfo;
            zeno::parseUpdateInfo(pNode->get_customui(), updateInfo);
            pNode->update_editparams(updateInfo, true);
        }
    }
    if (cls == "SubInput") {
        subinput_nodes.insert(uuid);
    }
    if (cls == "SubOutput") {
        suboutput_nodes.insert(uuid);
    }

    pNode->set_pos(pos);
    pNode->mark_dirty(true);
    m_name2uuid[name] = uuid;
    m_nodes.insert(std::make_pair(uuid, std::move(upNode)));

    CALLBACK_NOTIFY(createNode, name, pNode)
    return pNode;
}

Graph* Graph::addSubnetNode(std::string const& id) {
    //deprecated:
    return nullptr;
}

std::set<std::string> Graph::getSubInputs()
{
    std::set<std::string> inputs;
    for (auto&[name, uuid] : m_name2uuid)
    {
        if (subinput_nodes.find(uuid) != subinput_nodes.end())
            inputs.insert(name);
    }
    return inputs;
}

std::set<std::string> Graph::getSubOutputs()
{
    std::set<std::string> outputs;
    for (auto& [name, uuid] : m_name2uuid)
    {
        if (suboutput_nodes.find(uuid) != suboutput_nodes.end())
            outputs.insert(name);
    }
    return outputs;
}

bool Graph::hasNode(std::string const& uuid_node_path) {
    auto spNode = zeno::getSession().getNodeByUuidPath(uuid_node_path);
    if (!spNode) return false;
    return spNode->getThisGraph() == this;
}

NodeImpl* Graph::getNode(std::string const& name) {
    if (m_name2uuid.find(name) == m_name2uuid.end()) {
        return nullptr;
    }
    const std::string& uuid = m_name2uuid[name];
    return safe_at(m_nodes, uuid, "").get();
}

NodeImpl* Graph::getNodeByUuidPath(ObjPath path) {
    if (path.empty())
        return nullptr;

    int idx = path.find('/');
    std::string uuid = path.substr(0, idx);
    auto it = m_nodes.find(uuid);
    if (it == m_nodes.end()) {
        return nullptr;
    }
    if (idx != std::string::npos)
    {
        path = path.substr(idx + 1, path.size() - idx);
        if (SubnetNode* subnetNode = dynamic_cast<SubnetNode*>(it->second.get()))
        {
            auto spGraph = subnetNode->get_subgraph();
            if (spGraph)
                return spGraph->getNodeByUuidPath(path);
            else
                return nullptr;
        }
    }
    return it->second.get();
}

std::shared_ptr<Graph> Graph::_getGraphByPath(std::vector<std::string> items)
{
    if (items.empty())
        return shared_from_this();

    std::string currname = items[0];
    items.erase(items.begin());
    if (m_name == currname) {
        return _getGraphByPath(items);
    }

    if (m_name2uuid.find(currname) == m_name2uuid.end())
    {
        if (currname == ".") {
            return _getGraphByPath(items);
        }
        else if (currname == "..") {
            if (NodeImpl* pSubnetImpl = getParentSubnetNode()) {
                SubnetNode* parentNode = getSubnetNode(pSubnetImpl);
                auto parentG = pSubnetImpl->getGraph();
                return parentG->_getGraphByPath(items);
            }
        }
        return nullptr;
    }

    std::string uuid = m_name2uuid[currname];
    auto it = m_nodes.find(uuid);
    if (it == m_nodes.end()) {
        return nullptr;
    }

    if (auto subnetNode = getSubnetNode(it->second.get()))
    {
        auto spGraph = subnetNode->get_subgraph();
        if (spGraph)
            return spGraph->_getGraphByPath(items);
        else
            return nullptr;
    }
    return nullptr;
}

std::shared_ptr<Graph> Graph::getGraphByPath(const std::string& pa)
{
    std::string path = pa;
    if (path.empty())
        return nullptr;

    auto pathitems = split_str(pa, '/', false);
    return _getGraphByPath(pathitems);
}

zeno::NodeImpl* Graph::getNodeByPath(const std::string& pa)
{
    std::string path = pa;
    if (path.empty())
        return nullptr;

    auto pathitems = split_str(pa, '/', false);
    if (pathitems.empty())
        return nullptr;

    std::string nodename = pathitems.back();
    pathitems.pop_back();
    auto spGraph = _getGraphByPath(pathitems);
    if (!spGraph)
        return nullptr;
    return spGraph->getNode(nodename);
}

NodeImpl* Graph::getParentSubnetNode() const {
    return m_parSubnetNode;
}

void Graph::initParentSubnetNode(NodeImpl* pSubnetNode) {
    m_parSubnetNode = pSubnetNode;
}

std::vector<NodeImpl*> Graph::getNodesByClass(const std::string& cls)
{
    std::vector<NodeImpl*> nodes;
    auto iter = node_set.find(cls);
    if (iter != node_set.end()) {
        for (auto name : iter->second) {
            nodes.push_back(getNode(name));
        }
    }
    return nodes;
}

std::map<std::string, NodeImpl*> Graph::getNodes() const {
    std::map<std::string, NodeImpl*> nodes;
    for (auto& [uuid, node] : m_nodes) {
        nodes.insert(std::make_pair(node->get_name(), node.get()));
    }
    return nodes;
}

float Graph::statistic_cpu_used() const {
    float total_time = 0;
    for (auto& [uuid, node] : m_nodes) {
        //dirty, foreach的情况
        NodeRunStatus status = node->get_run_status();
        if (status == Node_DirtyReadyToRun) {
            float time = node->time();
            total_time += time;
        }
    }
    return total_time;
}

std::set<std::string> Graph::get_viewnodes() const {
    return m_viewnodes;
}

GraphData Graph::exportGraph() const {
    GraphData graph;
    graph.name = m_name;
    if ("main" == graph.name) {
        graph.type = Subnet_Main;
    }
    else {
        graph.type = Subnet_Normal;
        graph.links = exportLinks();
    }

    for (auto& [uuid, node] : m_nodes) {
        zeno::NodeData nodeinfo = node->exportInfo();
        graph.nodes.insert(std::make_pair(node->get_name(), nodeinfo));
    }
    return graph;
}

LinksData Graph::exportLinks() const
{
    LinksData links;
    for (auto& [uuid, node] : m_nodes) {
        zeno::NodeData nodeinfo = node->exportInfo();
        const PrimitiveParams& params = customUiToParams(nodeinfo.customUi.inputPrims);
        for (ParamPrimitive param : params) {
            links.insert(links.end(), param.links.begin(), param.links.end());
        }
        for (ParamObject param : nodeinfo.customUi.inputObjs) {
            links.insert(links.end(), param.links.begin(), param.links.end());
        }
    }
    return links;
}

std::string Graph::getName() const {
    if (m_parSubnetNode) {
        return m_parSubnetNode->get_name();
    }
    return m_name;
}

void Graph::setName(const std::string& na) {
    m_name = na;
}

bool Graph::removeNode(std::string const& name) {
    auto it = m_name2uuid.find(name);
    std::string uuid = safe_at(m_name2uuid, name, "get uuid when calling removeNode");
    auto spNode = safe_at(m_nodes, uuid, "").get();

    //remove links first
    std::vector<EdgeInfo> remLinks = spNode->getLinks();
    for (auto edge : remLinks) {
        removeLink(edge);
    }

    //再通知前端删节点
    CALLBACK_NOTIFY(removeNode, name)

    spNode->mark_dirty_objs();

    const std::string nodecls = spNode->get_nodecls();

    spNode->on_node_about_to_remove();

    node_set[nodecls].erase(name);
    m_nodes.erase(uuid);

    frame_nodes.erase(uuid);
    subnet_nodes.erase(uuid);
    asset_nodes.erase(uuid);
    m_viewnodes.erase(name);
    m_name2uuid.erase(name);

    return true;
}

bool zeno::Graph::isLinkValid(const EdgeInfo& edge)
{
    NodeImpl* outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    NodeImpl* inNode = getNode(edge.inNode);
    if (!inNode)
        return false;

    bool bExist = false;
    bool bOutputPrim = outNode->isPrimitiveType(false, edge.outParam, bExist);
    bool bInputPrim = inNode->isPrimitiveType(true, edge.inParam, bExist);

    if (!bExist) {
        zeno::log_warn("no exist param for edge.");
        return false;
    }
    if (bInputPrim != bOutputPrim) {
        zeno::log_warn("link type no match.");
        return false;
    }

    bool bInWildcard = false;
    bool bOutWildcard = false;
    SocketType outSocketType;
    ParamType outParamType;
    outNode->getParamTypeAndSocketType(edge.outParam, bOutputPrim, false, outParamType, outSocketType, bOutWildcard);
    SocketType inSocketType;
    ParamType inParamType;
    inNode->getParamTypeAndSocketType(edge.inParam, bOutputPrim, true, inParamType, inSocketType, bInWildcard);

    if (bOutWildcard || bInWildcard) {
        return true;
    } else if ((!bInputPrim && isSubnetInputOutputParam(inNode, edge.inParam)) || (!bOutputPrim && isSubnetInputOutputParam(outNode, edge.outParam))) {
        return true;
    }

    if (inParamType != outParamType)
    {
        NodeDataGroup outGroup = bOutputPrim ? Role_OutputPrimitive : Role_OutputObject;
        NodeDataGroup inGroup = bInputPrim ? Role_InputPrimitive : Role_InputObject;
        if (outParamTypeCanConvertInParamType(outParamType, inParamType, outGroup, inGroup)) {
        }
        else {
            zeno::log_warn("param type no match.");
            return false;
        }
    }

    return true;
}

bool Graph::addLink(const EdgeInfo& edge) {
    CORE_API_BATCH

    if (!isLinkValid(edge))
        return false;

    NodeImpl* outNode = getNode(edge.outNode);
    NodeImpl* inNode = getNode(edge.inNode);

    bool bExist = false;
    bool bOutputPrim = outNode->isPrimitiveType(false, edge.outParam, bExist);
    bool bInputPrim = inNode->isPrimitiveType(true, edge.inParam, bExist);

    EdgeInfo adjustEdge = edge;

    bool bConnectWithKey = false;
    adjustEdge.inKey = edge.inKey;

    if (bInputPrim)
    {
        ParamPrimitive inParam = inNode->get_input_prim_param(edge.inParam, &bExist);
        if (!bExist)
            return false;
        ParamPrimitive outParam = outNode->get_output_prim_param(edge.outParam, &bExist);
        if (!bExist)
            return false;
        if (inParam.type == gParamType_ListOfMat4)
        {
            if (outParam.type == gParamType_Matrix4)
            {
                bConnectWithKey = true;
                std::vector<EdgeInfo> inParamLinks = inParam.links;

                //要先检查一下已有的边是不是直连，如果是，要删掉
                if (inParam.links.size() == 1 && inParam.links[0].inKey.empty()) {
                    removeLinks(inNode->get_name(), true, edge.inParam);
                    inParam.links.clear();
                }

                std::set<std::string> ss;
                for (const EdgeInfo& spLink : inParam.links) {
                    ss.insert(spLink.inKey);
                }

                if (adjustEdge.inKey.empty())
                    adjustEdge.inKey = "obj0";

                int i = 0;
                while (ss.find(adjustEdge.inKey) != ss.end()) {
                    i++;
                    adjustEdge.inKey = "obj" + std::to_string(i);
                }
            }
        }
    }
    else
    {
        ParamObject inParam = inNode->get_input_obj_param(edge.inParam);
        ParamObject outParam = outNode->get_output_obj_param(edge.outParam);
        if (inParam.type == gParamType_Dict || inParam.type == gParamType_List) {
            std::vector<EdgeInfo> inParamLinks = inParam.links;
            if (inParamLinks.size() <= 1)
            {
                //像IObject这种既可能是List也可能是子元素的，用coreapi没法区分，只能在外部指定key决定
                //是直连还是子元素
                if (inParam.type == outParam.type) {
                    bConnectWithKey = false;
                }
                else {
                    bConnectWithKey = true;
                }
            }
            else {
                bConnectWithKey = true;
            }

            if (!edge.inKey.empty()) {
                //如果指定了inKey，无论外面是不是List，必须作为子元素
                bConnectWithKey = true;
            }
            else if (inParam.type == outParam.type || outParam.type == gParamType_IObject) {
                //连一个list/obj进来，而且没有指定key，就认为是直连
                bConnectWithKey = false;
            }

            if (bConnectWithKey) {
                //要先检查一下已有的边是不是直连，如果是，要删掉
                if (inParam.links.size() == 1 && inParam.links[0].inKey.empty()) {
                    removeLinks(inNode->get_name(), true, edge.inParam);
                    inParam.links.clear();
                }

                std::set<std::string> ss;
                for (const EdgeInfo& spLink : inParam.links) {
                    ss.insert(spLink.inKey);
                }

                if (adjustEdge.inKey.empty())
                    adjustEdge.inKey = "obj0";

                int i = 0;
                while (ss.find(adjustEdge.inKey) != ss.end()) {
                    i++;
                    adjustEdge.inKey = "obj" + std::to_string(i);
                }
            }
        }
    }

    if (!bConnectWithKey) {
        removeLinks(inNode->get_name(), true, edge.inParam);
    } else {
        std::vector<EdgeInfo> links = inNode->getLinksByParam(true, edge.inParam);
        for (auto link : links) {
            if (link.inNode == adjustEdge.inNode && link.outNode == adjustEdge.outNode &&
                link.inParam == adjustEdge.inParam && link.outParam == adjustEdge.outParam) {
                adjustEdge.inKey = link.inKey;
                removeLink(link);
                break;
            }
        }
    }

    assert(bInputPrim == bOutputPrim);
    if (bInputPrim) {
        std::shared_ptr<PrimitiveLink> spLink = std::make_shared<PrimitiveLink>();
        spLink->fromkey = adjustEdge.outKey;
        spLink->tokey = adjustEdge.inKey;
        outNode->init_primitive_link(false, edge.outParam, spLink, edge.targetParam);
        inNode->init_primitive_link(true, edge.inParam, spLink, edge.targetParam);
        adjustEdge.bObjLink = false;
    }
    else {
        std::shared_ptr<ObjectLink> spLink = std::make_shared<ObjectLink>();
        spLink->fromkey = adjustEdge.outKey;
        spLink->tokey = adjustEdge.inKey;
        outNode->init_object_link(false, edge.outParam, spLink, edge.targetParam);
        inNode->init_object_link(true, edge.inParam, spLink, edge.targetParam);
        adjustEdge.bObjLink = true;
    }

    inNode->mark_dirty(true);

    SocketType outSocketType;
    ParamType outParamType;
    bool bInWildcard = false;
    bool bOutWildcard = false;

    outNode->getParamTypeAndSocketType(edge.outParam, bOutputPrim, false, outParamType, outSocketType, bOutWildcard);
    SocketType inSocketType;
    ParamType inParamType;
    inNode->getParamTypeAndSocketType(edge.inParam, bOutputPrim, true, inParamType, inSocketType, bInWildcard);
    inNode->on_link_added_removed(true, edge.inParam, true);
    outNode->on_link_added_removed(false, edge.outParam, true);

    CALLBACK_NOTIFY(addLink, adjustEdge);
    return true;
}

void Graph::update_load_info(const std::string& nodecls, bool bDisable) {
    auto iter = node_set.find(nodecls);
    if (iter != node_set.end()) {
        for (auto name : iter->second) {
            auto iter2 = m_name2uuid.find(name);
            assert(iter2 != m_name2uuid.end());
            auto iter3 = m_nodes.find(iter2->second);
            assert(iter3 != m_nodes.end());
            iter3->second->update_load_info(bDisable);
        }
    }
    //要递归遍历所有子图
    for (const std::string& subnetnode : subnet_nodes) {
        auto pNodeImpl = m_nodes[subnetnode].get();
        auto spSubnetNode = dynamic_cast<SubnetNode*>(pNodeImpl);
        assert(spSubnetNode);
        spSubnetNode->get_subgraph()->update_load_info(nodecls, bDisable);
    }
}

bool Graph::removeLink(const EdgeInfo& edge) {
    CORE_API_BATCH

    CALLBACK_NOTIFY(removeLink, edge)

    NodeImpl* outNode = getNode(edge.outNode);
    if (!outNode)
        return false;

    NodeImpl* inNode = getNode(edge.inNode);
    if (!inNode)
        return false;

    //pre checking for param.
    bool bExist = false;
    bool bPrimType = outNode->isPrimitiveType(/*bool bInput*/false, edge.outParam, bExist);
    if (!bExist)
        return false;
    bool bPrimType2 = inNode->isPrimitiveType(/*bool bInput*/true, edge.inParam, bExist);
    if (!bExist && bPrimType != bPrimType2)
        return false;

    outNode->removeLink(false, edge);
    inNode->removeLink(true, edge);
    inNode->mark_dirty(true);

    SocketType inSocketType;
    ParamType inParamType;
    bool bInWildcard = false;
    bool bOutWildcard = false;
    inNode->getParamTypeAndSocketType(edge.inParam, bPrimType, true, inParamType, inSocketType, bInWildcard);
    resetWildCardParamsType(bInWildcard, inNode, edge.inParam, bPrimType, true);
    SocketType outSocketType;
    ParamType outParamType;
    outNode->getParamTypeAndSocketType(edge.outParam, bPrimType, false, outParamType, outSocketType, bOutWildcard);
    resetWildCardParamsType(bOutWildcard, outNode, edge.outParam, bPrimType, false);

    if (!bPrimType2) {
        const ParamObject& inParam = inNode->get_input_obj_param(edge.inParam);
        if (inParam.type == gParamType_List || inParam.type == gParamType_Dict) {
            std::vector<EdgeInfo> inParamLinks = inParam.links;
            if (inParamLinks.size() == 1) {
                if (auto node = getNode(inParamLinks[0].outNode)) {
                    ParamObject existOneParam = node->get_output_obj_param(inParamLinks[0].outParam);
                    if (existOneParam.type == inParam.type) {
                        updateLink(inParamLinks[0], false, inParamLinks[0].inKey, "");
                    }
                }
            }
        }
    }

    inNode->on_link_added_removed(true, edge.inParam, false);
    outNode->on_link_added_removed(false, edge.outParam, false);

    return true;
}

bool Graph::removeLinks(const std::string nodename, bool bInput, const std::string paramname)
{
    CORE_API_BATCH

    NodeImpl* spNode = getNode(nodename);
    std::vector<EdgeInfo> links = spNode->getLinksByParam(bInput, paramname);
    for (auto link : links)
        removeLink(link);

    CALLBACK_NOTIFY(removeLinks, nodename, bInput, paramname)
    return true;
}

bool Graph::updateLink(const EdgeInfo& edge, bool bInput, const std::string oldkey, const std::string newkey)
{
    CORE_API_BATCH

    NodeImpl* outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    NodeImpl* inNode = getNode(edge.inNode);
    if (!inNode)
        return false;

    bool bExist = false;
    bool bOutputPrim = outNode->isPrimitiveType(false, edge.outParam, bExist);
    if (!bExist)
        return false;
    bool bInputPrim = inNode->isPrimitiveType(true, edge.inParam, bExist);
    if (!bExist)
        return false;
    if (bInputPrim != bOutputPrim)
        return false;
    return inNode->updateLinkKey(true, edge, oldkey, newkey);
}

bool Graph::moveUpLinkKey(const EdgeInfo& edge, bool bInput, const std::string keyName)
{
    CORE_API_BATCH
    NodeImpl* outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    NodeImpl* inNode = getNode(edge.inNode);
    if (!inNode)
        return false;
    return moveUpLinkKey(edge, bInput, keyName);
}

}
