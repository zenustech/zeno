#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
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
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <zeno/core/CoreParam.h>
#include <zeno/utils/uuid.h>
#include <zeno/utils/helper.h>
#include <iostream>
#include <regex>
#include <zeno/core/GlobalVariable.h>
#include <zeno/core/typeinfo.h>
#include "zeno_types/reflect/reflection.generated.hpp"


namespace zeno {

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{}

ZENO_API Graph::Graph(const std::string& name, bool bAssets) : m_name(name), m_bAssets(bAssets) {
    
}

ZENO_API Graph::~Graph() {

}

ZENO_API zany Graph::getNodeInput(std::string const& sn, std::string const& ss) const {
    //todo: deprecated
    auto node = safe_at(m_nodes, sn, "node name").get();
    return node->get_input(ss);
}

ZENO_API void Graph::clearNodes() {
    m_nodes.clear();
}

ZENO_API void Graph::addNode(std::string const &cls, std::string const &id) {
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

ZENO_API Graph *Graph::getSubnetGraph(std::string const & node_name) const {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    auto node = static_cast<SubnetNode *>(safe_at(m_nodes, uuid, "node name").get());
    return node->subgraph.get();
}

ZENO_API void Graph::completeNode(std::string const &node_name) {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    safe_at(m_nodes, uuid, "node name")->doComplete();
}

void Graph::foreachApply(INode* foreach_end) {
    std::string foreach_begin_path = zeno::reflect::any_cast<std::string>(foreach_end->get_defl_value("ForEachBegin Path"));
    auto foreach_begin = getNode(foreach_begin_path);
    if (!foreach_begin) {
        throw makeError<KeyError>("foreach_begin_path", "the path of foreach_begin_path is not exist");
    }

    for (foreach_end->reset_forloop_settings(); foreach_end->is_continue_to_run(); foreach_end->increment())
    {
        foreach_begin->mark_dirty(true);
        foreach_end->doApply();
    }
    foreach_end->registerObjToManager();
    //foreach_end->reportStatus(false, Node_RunSucceed);
}

void Graph::timeshiftApply(INode* timeshiftNode)
{
    if (timeshiftNode) {
        int oldFrame = getSession().globalState->getFrameId();
        scope_exit sp([&oldFrame] { getSession().globalState->updateFrameId(oldFrame); });
        //get offset
        auto defl = timeshiftNode->get_input_prim_param("offset").defl;
        zeno::PrimVar offset = defl.has_value() ? zeno::reflect::any_cast<zeno::PrimVar>(defl) : 0;
        int newFrame = oldFrame + std::get<int>(offset);
        //clamp
        auto startFrameDefl = timeshiftNode->get_input_prim_param("start frame").defl;
        int globalStartFrame = getSession().globalState->getStartFrame();
        int startFrame = startFrameDefl.has_value() ? std::get<int>(zeno::reflect::any_cast<PrimVar>(startFrameDefl)) : globalStartFrame;
        auto endFrameDefl = timeshiftNode->get_input_prim_param("end frame").defl;
        int globalEndFrame = getSession().globalState->getEndFrame();
        int endFrame = endFrameDefl.has_value() ? std::get<int>(zeno::reflect::any_cast<PrimVar>(endFrameDefl)) : globalEndFrame;
        auto clampDefl = timeshiftNode->get_input_prim_param("clamp").defl;
        std::string clamp = clampDefl.has_value() ? zeno::reflect::any_cast<std::string>(clampDefl) : "None";
        if (startFrame > endFrame) {
            startFrame = globalStartFrame;
            endFrame = globalEndFrame;
        }
        if (clamp == "Clamp to First") {
            newFrame = newFrame < startFrame ? startFrame : newFrame;
        }
        else if (clamp == "Clamp to Last") {
            newFrame = newFrame > endFrame ? endFrame : newFrame;
        }
        else if (clamp == "Clamp to Both") {
            if (newFrame < startFrame) {
                newFrame = startFrame;
            }
            else if (newFrame > endFrame) {
                newFrame = endFrame;
            }
        }
        getSession().globalState->updateFrameId(newFrame);
        //propaget dirty
        std::shared_ptr<INode> spnode = safe_at(m_nodes, timeshiftNode->get_uuid(), "node name");
        propagateDirty(spnode, "$F");
        timeshiftNode->doApply();
    }
}

ZENO_API bool Graph::applyNode(std::string const &node_name) {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    auto node = safe_at(m_nodes, uuid, "node name").get();

    if (this->visited.find(uuid) != this->visited.end()) {
        throw makeError<UnimplError>("cycle reference occurs!");
    }

    this->visited.insert(uuid);
    scope_exit sp([=] {this->visited.erase(uuid); });

    GraphException::translated([&] {
        std::string nodecls = node->get_nodecls();
        if ("TimeShift" == nodecls) {
            timeshiftApply(node);
        } else if ("ForEachEnd" == node->get_nodecls() && node->is_dirty()) {
            foreachApply(node);
        }
        else {
            node->doApply();
        }
    }, node);

    return true;
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &nodes) {
    for (auto const& node_name: nodes) {
        applyNode(node_name);
    }
}

ZENO_API void Graph::runGraph() {
    log_debug("{} nodes to exec", m_viewnodes.size());
    applyNodes(m_viewnodes);
}

void Graph::onNodeParamUpdated(PrimitiveParam* spParam, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
    auto spNode = spParam->m_wpNode.lock();
    assert(spNode);
    {   //检测param依赖全局变量,先remove再parse
        const std::string& uuid = spNode->get_uuid();
        frame_nodes.erase(uuid);
        assert(spParam);
        parseNodeParamDependency(spParam, new_value);
    }
}

void Graph::parseNodeParamDependency(PrimitiveParam* spParam, zeno::reflect::Any& new_value)
{
    auto spNode = spParam->m_wpNode.lock();
    assert(spNode);
    assert(spParam->type == Param_Wildcard || spParam->defl.has_value());
    const std::string& uuid = spNode->get_uuid();
    if (gParamType_String == spParam->type)
    {
        std::string defl = zeno::reflect::any_cast<std::string>(spParam->defl);
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
        auto viewnodes = m_viewnodes;
        for (auto nodename : viewnodes) {
            auto spNode = getNode(nodename);
            spNode->set_view(false);
        }
        m_viewnodes.insert(node);
    }
    else {
        m_viewnodes.erase(node);
        //TODO: update objsmanager to hide objs.
    }
}

ZENO_API void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    //safe_at(nodes, dn, "node name")->inputBounds[ds] = std::pair(sn, ss);
}

ZENO_API void Graph::setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    //todo: deprecated.
    //safe_at(nodes, id, "node name")->inputs[par] = val;
}

ZENO_API void Graph::setKeyFrame(std::string const &id, std::string const &par, zany const &val) {
    //todo: deprecated.
    /*
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->kframes.insert(par);
    */
}

ZENO_API void Graph::setFormula(std::string const &id, std::string const &par, zany const &val) {
    //todo: deprecated.
    /*
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->formulas.insert(par);
    */
}


ZENO_API std::map<std::string, zany> Graph::callSubnetNode(std::string const &id,
        std::map<std::string, zany> inputs) const {
    //todo: deprecated.
    return std::map<std::string, zany>();
}

ZENO_API std::map<std::string, zany> Graph::callTempNode(std::string const &id,
        std::map<std::string, zany> inputs) {

    //DEPRECARED.
    return {};
#if 0
    auto cl = safe_at(getSession().nodeClasses, id, "node class name").get();
    const std::string& name = generateUUID();
    auto se = cl->new_instance(shared_from_this(), name);
    se->directly_setinputs(inputs);
    se->doOnlyApply();
    return se->getoutputs();
#endif
}

ZENO_API void Graph::addNodeOutput(std::string const& id, std::string const& par) {
    // add "dynamic" output which is not descriped by core.
    //todo: deprecated.
    //safe_at(nodes, id, "node name")->outputs[par] = nullptr;
}

ZENO_API void Graph::setNodeParam(std::string const &id, std::string const &par,
    std::variant<int, float, std::string, zany> const &val) {
    auto parid = par + ":";
    std::visit([&] (auto const &val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, zany>) {
            setNodeInput(id, parid, val);
        } else {
            setNodeInput(id, parid, objectFromLiterial(val));
        }
    }, val);
}

ZENO_API DirtyChecker &Graph::getDirtyChecker() {
    if (!dirtyChecker) 
        dirtyChecker = std::make_unique<DirtyChecker>();
    return *dirtyChecker;
}

ZENO_API void Graph::init(const GraphData& graph) {
    auto& sess = getSession();
    sess.setApiLevelEnable(false);
    zeno::scope_exit([&]() {
        sess.setApiLevelEnable(true);
    });

    m_name = graph.name;
    //import nodes first.
    for (const auto& [name, node] : graph.nodes) {
        bool bAssets = node.asset.has_value();
        std::shared_ptr<INode> spNode = createNode(node.cls, name, bAssets);
        spNode->init(node);
        if (node.cls == "SubInput") {
            //TODO
        }
        else if (node.cls == "SubOutput") {
            //TODO
        }
        else if (node.cls == "Group") {
            if (node.group.has_value())
            {
                spNode->update_param("title", node.group->title);
                spNode->update_param("background", node.group->background);
                spNode->update_param("size", node.group->sz);
                spNode->update_param("items", join_str(node.group->items, ","));
            }
        }
        //Compatible with older versions
        else if (node.cls == "MakeHeatmap")
        {
            std::string color;
            int nres = 0;
            const PrimitiveParams& primparams = customUiToParams(node.customUi.inputPrims);
            for (const auto& input : primparams)
            {
                if (input.name == "_RAMPS")
                {
                    color = zeno_get<std::string>(input.defl);
                }
                else if (input.name == "nres")
                {
                    nres = zeno_get<int>(input.defl);
                }
            }
            if (!color.empty() && nres > 0)
            {
                std::regex pattern("\n");
                std::string fmt = "\\n";
                color = std::regex_replace(color, pattern, fmt);
                std::string json = "{\"nres\": " + std::to_string(nres) + ", \"color\":\"" + color + "\"}";
                spNode->update_param("heatmap", json);
            }
        }
        else if (zeno::isDerivedFromSubnetNodeName(node.cls))
        {
            if (std::shared_ptr<zeno::SubnetNode> sbn = std::dynamic_pointer_cast<zeno::SubnetNode>(spNode))
                sbn->setCustomUi(node.customUi);
        }
    }
    //import edges
    for (const auto& link : graph.links) {
        if (!isLinkValid(link))
            continue;
        std::shared_ptr<INode> outNode = getNode(link.outNode);
        std::shared_ptr<INode> inNode = getNode(link.inNode);

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

    for (const auto& [nodename, refparams] : graph.references) {
        std::shared_ptr<INode> refNode = getNode(nodename);
        const auto& uuidpath = refNode->get_uuid_path();
        for (auto paramname : refparams) {
            refNode->constructReference(paramname);
        }
    }
}

void Graph::markDirtyWhenFrameChanged()
{
    for (const std::string& uuid : frame_nodes) {
        if (!m_nodes[uuid]->isInDopnetwork()) {//不在dop节点中才markDirty
        m_nodes[uuid]->mark_dirty(true);
    }
    }
    std::set<std::string> nodes = subnet_nodes;
    nodes.insert(asset_nodes.begin(), asset_nodes.end());
    for (const std::string& uuid : nodes) {
        auto spSubnetNode = std::dynamic_pointer_cast<SubnetNode>(m_nodes[uuid]);
        spSubnetNode->subgraph->markDirtyWhenFrameChanged();
    }
}

void Graph::markDirtyAll()
{
    for (const auto& [uuid, node] : m_nodes) {
        node->mark_dirty(true);
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
    while (true) {
        std::string new_name = tempName + std::to_string(i++);
        if (nodes.find(new_name) == nodes.end()) {
            nodes.insert(new_name);
            return new_name;
        }
    }
    return "";
}

void Graph::updateWildCardParamTypeRecursive(std::shared_ptr<Graph> spCurrGarph, std::shared_ptr<INode> spNode, std::string paramName, bool bPrim, bool bInput, ParamType newtype)
{
    if (!spCurrGarph || !spNode)
        return;
    if (spNode->get_nodecls() == "SubOutput" || spNode->get_nodecls() == "SubInput") { //由子图内部传导出来
        spNode->update_param_type(paramName, bPrim, bInput, newtype);
        auto links = spNode->getLinksByParam(bInput, paramName);
        for (auto& link : links) {
            if (bInput) {}
                //updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.outNode), link.outParam, bPrim, !bInput, newtype);
            else {
                if (auto& innode = spCurrGarph->getNode(link.inNode)) {
                    ParamType paramType;
                    SocketType socketType;
                    innode->getParamTypeAndSocketType(link.inParam, bPrim, !bInput, paramType, socketType);
                    if (socketType == Socket_WildCard)
                        updateWildCardParamTypeRecursive(spCurrGarph, innode, link.inParam, bPrim, !bInput, newtype);
                    else if (paramType != newtype)
                        spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                }
            }
        }
        if (std::shared_ptr<Graph> graph = spNode->getGraph().lock()) {
            if (graph->optParentSubgNode.has_value()) {
                if (SubnetNode* parentSubgNode = graph->optParentSubgNode.value()) {
                    parentSubgNode->update_param_type(spNode->get_name(), bPrim, !bInput, newtype);
                    for (auto& link : parentSubgNode->getLinksByParam(!bInput, spNode->get_name())) {
                        if (std::shared_ptr<Graph> parentGraph = parentSubgNode->getGraph().lock()) {
                            auto const& inNode = parentGraph->getNode(link.inNode);
                            auto const& outNode = parentGraph->getNode(link.outNode);
                            ParamType inNodeParamType;
                            SocketType inNodeSocketType;
                            ParamType outNodeParamType;
                            SocketType outNodeSocketType;
                            inNode->getParamTypeAndSocketType(link.inParam, bPrim, true, inNodeParamType, inNodeSocketType);
                            outNode->getParamTypeAndSocketType(link.outParam, bPrim, false, outNodeParamType, outNodeSocketType);
                            if (inNodeParamType != outNodeParamType) {
                                if (inNodeSocketType != Socket_WildCard)
                                    parentGraph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                                else {
                                    if (bInput)
                                        updateWildCardParamTypeRecursive(parentGraph, inNode, link.inParam, bPrim, bInput, newtype);
                                    else {
                                        if (outNodeSocketType == Socket_WildCard)
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
    else if (zeno::isDerivedFromSubnetNodeName(spNode->get_nodecls())) {  //通过inputObj传入子图
        spNode->update_param_type(paramName, bPrim, bInput, newtype);
        if (std::shared_ptr<SubnetNode> subnet = std::dynamic_pointer_cast<SubnetNode>(spNode)) {
            for (auto& link : subnet->getLinksByParam(bInput, paramName)) {
                if (bInput) {}
                //updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.outNode), link.outParam, bPrim, !bInput, newtype);
                else {
                    if (auto& innode = spCurrGarph->getNode(link.inNode)) {
                        ParamType paramType;
                        SocketType socketType;
                        innode->getParamTypeAndSocketType(link.inParam, bPrim, !bInput, paramType, socketType);
                        if (socketType == Socket_WildCard)
                            updateWildCardParamTypeRecursive(spCurrGarph, spCurrGarph->getNode(link.inNode), link.inParam, bPrim, !bInput, newtype);
                        else if (paramType != newtype)
                            spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                    }
                }
            }
            if (auto innerNode = subnet->subgraph->getNode(paramName)) {
                std::vector<std::string> inparamNames;
                if (bInput) {
                    for (auto& param : innerNode->get_output_object_params())
                        inparamNames.emplace_back(param.name);
                }
                else {
                    if (bPrim)
                        for (auto& param: innerNode->get_input_primitive_params())
                            inparamNames.emplace_back(param.name);
                    else
                        for (auto& param : innerNode->get_input_object_params())
                            inparamNames.emplace_back(param.name);
                }
                for (auto& name: inparamNames) {
                    innerNode->update_param_type(name, bPrim, !bInput, newtype);
                    for (auto& link: innerNode->getLinksByParam(!bInput, name)) {
                        auto const& inNode = subnet->subgraph->getNode(link.inNode);
                        auto const& outNode = subnet->subgraph->getNode(link.outNode);
                        ParamType inNodeParamType;
                        SocketType inNodeSocketType;
                        ParamType outNodeParamType;
                        SocketType outNodeSocketType;
                        inNode->getParamTypeAndSocketType(link.inParam, bPrim, true, inNodeParamType, inNodeSocketType);
                        outNode->getParamTypeAndSocketType(link.outParam, bPrim, false, outNodeParamType, outNodeSocketType);
                        if (inNodeParamType != outNodeParamType) {
                            if (inNodeSocketType != Socket_WildCard)
                                subnet->subgraph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                            else {
                                if (bInput)
                                    updateWildCardParamTypeRecursive(subnet->subgraph, inNode, link.inParam, bPrim, bInput, newtype);
                                else {
                                    if (outNodeSocketType == Socket_WildCard)
                                        updateWildCardParamTypeRecursive(subnet->subgraph, outNode, link.outParam, bPrim, bInput, newtype);
                                    else
                                        subnet->subgraph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                                }
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
            for (auto& link : spNode->getLinksByParam(param.second, param.first)) {      //有其他边连接这个参数，类型不同则删除
                std::shared_ptr<INode> otherNodeLinkToThis = param.second ? spCurrGarph->getNode(link.outNode) : spCurrGarph->getNode(link.inNode);
                if (otherNodeLinkToThis) {
                    if (param.second) { //是输入
                        ParamType paramType;
                        SocketType socketType;
                        otherNodeLinkToThis->getParamTypeAndSocketType(link.outParam, bPrim, false, paramType, socketType);
                        if (paramType != newtype) {
                            if (socketType == Socket_WildCard)
                                updateWildCardParamTypeRecursive(spCurrGarph, otherNodeLinkToThis, link.outParam, bPrim, false, newtype);
                            else
                                spCurrGarph->removeLinkWhenUpdateWildCardParam(link.outNode, link.inNode, link);
                        }
                    }
                    else {
                        ParamType paramType;
                        SocketType socketType;
                        otherNodeLinkToThis->getParamTypeAndSocketType(link.inParam, bPrim, true, paramType, socketType);
                        if (paramType != newtype) {
                            if (socketType == Socket_WildCard)
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
    std::shared_ptr<INode> outNode = getNode(outNodeName);
    std::shared_ptr<INode> inNode = getNode(inNodeName);
    if (!outNode || !inNode)
        return;
    outNode->removeLink(false, edge);
    inNode->removeLink(true, edge);
    inNode->mark_dirty(true);
    CALLBACK_NOTIFY(removeLink, edge)
}

void Graph::resetWildCardParamsType(SocketType& socketType, std::shared_ptr<INode>& node, const std::string& paramName, const bool& bPrimType, const bool& bInput)
{
    if (!node)
        return;
    std::function<bool(std::shared_ptr<Graph>, std::shared_ptr<INode>, std::string, bool, std::set<std::string>&)> linkedToSpecificType =
        [&linkedToSpecificType, this](std::shared_ptr<Graph> currGraph, std::shared_ptr<INode> node, std::string paramName, bool bPrimType, std::set<std::string>& visited)->bool {
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
                if (std::shared_ptr<Graph> graph = node->getGraph().lock()) {
                    if (graph->optParentSubgNode.has_value()) {
                        if (SubnetNode* parentSubgNode = graph->optParentSubgNode.value()) {
                            visited.insert(parentSubgNode->get_uuid() + node->get_uuid());
                            if (auto parentGraph = parentSubgNode->getGraph().lock()) {
                                for (auto& link : parentSubgNode->getLinksByParam(!param.second, node->get_name())) {
                                    const auto& node = parentGraph->getNode(param.second ? link.inNode : link.outNode);
                                    ParamType paramType;
                                    SocketType socketType;
                                    node->getParamTypeAndSocketType(param.second ? link.inParam : link.outParam, bPrimType, param.second, paramType, socketType);
                                    if (socketType != Socket_WildCard)
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
                if (std::shared_ptr<SubnetNode> subnet = std::dynamic_pointer_cast<SubnetNode>(node)) {
                    if (auto innerNode = subnet->subgraph->getNode(paramName)) {
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
                                auto node = subnet->subgraph->getNode(param.second ? link.inNode : link.outNode);
                                ParamType paramType;
                                SocketType socketType;
                                node->getParamTypeAndSocketType(param.second ? link.inParam : link.outParam, bPrimType, param.second, paramType, socketType);
                                if (socketType != Socket_WildCard)
                                    return true;
                                else {
                                    if (!visited.count(node->get_uuid() + (param.second ? link.inParam : link.outParam)))
                                        if (linkedToSpecificType(subnet->subgraph, node, param.second ? link.inParam : link.outParam, bPrimType, visited))
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
                    node->getParamTypeAndSocketType(param.second ? link.outParam : link.inParam, bPrimType, !param.second, paramType, socketType);
                    if (socketType != Socket_WildCard)
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
    if (socketType == Socket_WildCard) {
        if (linkedToSpecificType(shared_from_this(), node, paramName, bPrimType, std::set<std::string>()))
            return;
        updateWildCardParamTypeRecursive(shared_from_this(), node, paramName, bPrimType, bInput, bPrimType ? Param_Wildcard : Obj_Wildcard);
    }
}

ZENO_API bool Graph::isAssets() const
{
    return m_bAssets;
}

ZENO_API std::set<std::string> Graph::searchByClass(const std::string& name) const
{
    auto it = node_set.find(name);
    if (it == node_set.end())
        return {};
    return it->second;
}

ZENO_API std::string Graph::updateNodeName(const std::string oldName, const std::string newName)
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

    auto spNode = m_nodes[uuid];
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

    CALLBACK_NOTIFY(updateNodeName, oldName, name)
    return name;
}

ZENO_API void Graph::clear()
{
    m_nodes.clear();
    nodesToExec.clear();
    portalIns.clear();
    portals.clear();

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

    optParentSubgNode = std::nullopt;
    ctx.reset();
    dirtyChecker.reset();

    CALLBACK_NOTIFY(clear)
}

ZENO_API std::shared_ptr<INode> Graph::createNode(std::string const& cls, const std::string& orgin_name, bool bAssets, std::pair<float, float> pos)
{
    CORE_API_BATCH

    const std::string& name = generateNewName(cls, orgin_name, bAssets);

    std::string uuid;
    std::shared_ptr<INode> node;
    if (!bAssets) {
        auto& nodeClass = getSession().nodeClasses;
        std::string nodecls = cls;
        auto it = nodeClass.find(nodecls);
        if (it == nodeClass.end()) {
            nodecls = "DeprecatedNode";
        }

        auto cl = safe_at(getSession().nodeClasses, nodecls, "node class name").get();
        node = cl->new_instance(shared_from_this(), name);
        node->nodeClass = cl;
        uuid = node->get_uuid();
    }
    else {
        bool isCurrentGraphAsset = getSession().assets->isAssetGraph(shared_from_this());
        node = getSession().assets->newInstance(shared_from_this(), cls, name, isCurrentGraphAsset);
        uuid = node->get_uuid();
        asset_nodes.insert(uuid);
    }

    if (cls == "GetFrameNum") {
        frame_nodes.insert(uuid);
    }
    if (cls == "CameraNode") {   //相机相关节点和帧相关
        frame_nodes.insert(uuid);
    }
    if (zeno::isDerivedFromSubnetNodeName(cls)) {
        subnet_nodes.insert(uuid);
    }
    if (cls == "SubInput") {
        subinput_nodes.insert(uuid);
    }
    if (cls == "SubOutput") {
        suboutput_nodes.insert(uuid);
    }

    node->set_pos(pos);
    node->mark_dirty(true);
    m_name2uuid[name] = uuid;
    m_nodes[uuid] = node;

    CALLBACK_NOTIFY(createNode, name, node)
    return node;
}

ZENO_API Graph* Graph::addSubnetNode(std::string const& id) {
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

ZENO_API std::shared_ptr<INode> Graph::getNode(std::string const& name) {
    if (m_name2uuid.find(name) == m_name2uuid.end()) {
        return nullptr;
    }
    const std::string& uuid = m_name2uuid[name];
    return safe_at(m_nodes, uuid, "");
}

ZENO_API std::shared_ptr<INode> Graph::getNodeByUuidPath(ObjPath path) {
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
        //subnet
        if (std::shared_ptr<SubnetNode> subnetNode = std::dynamic_pointer_cast<SubnetNode>(it->second))
        {
            auto spGraph = subnetNode->subgraph;
            if (spGraph)
                return spGraph->getNodeByUuidPath(path);
            else
                return nullptr;
        }
    }
    return it->second;
}

std::shared_ptr<Graph> Graph::_getGraphByPath(std::vector<std::string> items)
{
    if (items.empty())
        return shared_from_this();

    std::string currname = items[0];
    items.erase(items.begin());
    if (m_name == "main") {
        if (currname == "main") {
            return _getGraphByPath(items);
        }
    }

    if (m_name2uuid.find(currname) == m_name2uuid.end())
    {
        if (currname == ".") {
            return _getGraphByPath(items);
        }
        else if (currname == "..") {
            //取parent graph.
            if (optParentSubgNode.has_value()) {
                SubnetNode* parentNode = optParentSubgNode.value();
                auto parentG = parentNode->getGraph().lock();
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

    if (std::shared_ptr<SubnetNode> subnetNode = std::dynamic_pointer_cast<SubnetNode>(it->second))
    {
        auto spGraph = subnetNode->subgraph;
        if (spGraph)
            return spGraph->_getGraphByPath(items);
        else
            return nullptr;
    }
    return nullptr;
}

ZENO_API std::shared_ptr<Graph> Graph::getGraphByPath(const std::string& pa)
{
    std::string path = pa;
    if (path.empty())
        return nullptr;

    auto pathitems = split_str(pa, '/', false);
    return _getGraphByPath(pathitems);
}

ZENO_API std::shared_ptr<INode> Graph::getNodeByPath(const std::string& pa)
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
    return spGraph->getNode(nodename);
}

ZENO_API std::map<std::string, std::shared_ptr<INode>> Graph::getNodes() const {
    std::map<std::string, std::shared_ptr<INode>> nodes;
    for (auto& [uuid, node] : m_nodes) {
        nodes.insert(std::make_pair(node->get_name(), node));
    }
    return nodes;
}

ZENO_API GraphData Graph::exportGraph() const {
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

ZENO_API LinksData Graph::exportLinks() const
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

ZENO_API std::string Graph::getName() const {
    if (optParentSubgNode.has_value()) {
        SubnetNode* pSubnetNode = optParentSubgNode.value();
        return pSubnetNode->get_name();
    }
    return m_name;
}

ZENO_API void Graph::setName(const std::string& na) {
    m_name = na;
}

ZENO_API bool Graph::removeNode(std::string const& name) {
    auto it = m_name2uuid.find(name);
    std::string uuid = safe_at(m_name2uuid, name, "get uuid when calling removeNode");
    auto spNode = safe_at(m_nodes, uuid, "");

    //remove links first
    std::vector<EdgeInfo> remLinks = spNode->getLinks();
    for (auto edge : remLinks) {
        removeLink(edge);
    }

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

    CALLBACK_NOTIFY(removeNode, name)
    return true;
}

bool zeno::Graph::isLinkValid(const EdgeInfo& edge)
{
    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    std::shared_ptr<INode> inNode = getNode(edge.inNode);
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

    SocketType outSocketType;
    ParamType outParamType;
    outNode->getParamTypeAndSocketType(edge.outParam, bOutputPrim, false, outParamType, outSocketType);
    SocketType inSocketType;
    ParamType inParamType;
    inNode->getParamTypeAndSocketType(edge.inParam, bOutputPrim, true, inParamType, inSocketType);

    if (outSocketType == zeno::Socket_WildCard || inSocketType == zeno::Socket_WildCard)
        return true;

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

ZENO_API bool Graph::addLink(const EdgeInfo& edge) {
    //如果输入端是dict/list，
    //外部调用者在调用此api时，有如下规则：
    //1.如果连进来的是dictlist，并且没有指定key，则认为是直接连此输入参数(类型为dictlist)
    //2.如果连进来的是dictlist，并且指定了key，则认为是连入dictlist内部并作为输入端的子成员。
    //3.如果连进来的是非dictlist，并且没有指定key，则认为是连入输入端dictlist并作为输入端的内部子成员。
    CORE_API_BATCH

    if (!isLinkValid(edge))
        return false;

    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    std::shared_ptr<INode> inNode = getNode(edge.inNode);

    bool bExist = false;
    bool bOutputPrim = outNode->isPrimitiveType(false, edge.outParam, bExist);
    bool bInputPrim = inNode->isPrimitiveType(true, edge.inParam, bExist);

    EdgeInfo adjustEdge = edge;

    bool bRemOldLinks = true, bConnectWithKey = false;
    adjustEdge.inKey = edge.inKey;

    if (!bInputPrim)
    {
        ParamObject inParam = inNode->get_input_obj_param(edge.inParam);
        ParamObject outParam = outNode->get_output_obj_param(edge.outParam);
        if (inParam.type == gParamType_Dict || inParam.type == gParamType_List) {
            std::vector<EdgeInfo> inParamLinks = inParam.links;
            if (inParamLinks.size() == 1) {
                if (auto node = getNode(inParamLinks[0].outNode)) {
                    ParamObject existOneParam = node->get_output_obj_param(inParamLinks[0].outParam);
                    if (existOneParam.type == inParam.type) {
                        updateLink(inParamLinks[0], false, inParamLinks[0].inKey, "obj0");
                        adjustEdge.inKey = "obj0";
                        inParam = inNode->get_input_obj_param(edge.inParam);
                    }
                }
                bRemOldLinks = false;
                bConnectWithKey = true;
            }else if (inParamLinks.size() < 1)
            {
                if (inParam.type == outParam.type) {
                    bRemOldLinks = true;
                    bConnectWithKey = false;
                }
                else {
                    bRemOldLinks = false;
                    bConnectWithKey = true;
                }
            }
            else {
                bRemOldLinks = false;
                bConnectWithKey = true;
            }
            if (bConnectWithKey) {
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
        if (inParam.socketType == Socket_Owning)
        {
            removeLinks(outNode->get_name(), false, edge.outParam);
        }
    }

    if (bRemOldLinks)
        removeLinks(inNode->get_name(), true, edge.inParam);

    assert(bInputPrim == bOutputPrim);
    if (bInputPrim) {
        std::shared_ptr<PrimitiveLink> spLink = std::make_shared<PrimitiveLink>();
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


    //加边之后，如果涉及wildCard的socket，传播wildcard类型
    SocketType outSocketType;
    ParamType outParamType;
    outNode->getParamTypeAndSocketType(edge.outParam, bOutputPrim, false, outParamType, outSocketType);
    SocketType inSocketType;
    ParamType inParamType;
    inNode->getParamTypeAndSocketType(edge.inParam, bOutputPrim, true, inParamType, inSocketType);
    if (outSocketType == zeno::Socket_WildCard && inSocketType == zeno::Socket_WildCard)
    {
        if (outParamType != inParamType) {
            ParamType newType;
            if (edge.targetParam == edge.outParam) {
                if (inParamType == Param_Wildcard || inParamType == Obj_Wildcard)
                    updateWildCardParamTypeRecursive(shared_from_this(), inNode, edge.inParam, bOutputPrim, true, outParamType);
                else
                    updateWildCardParamTypeRecursive(shared_from_this(), outNode, edge.outParam, bOutputPrim, false, inParamType);
            }
            else {
                if (outParamType == Param_Wildcard || outParamType == Obj_Wildcard)
                    updateWildCardParamTypeRecursive(shared_from_this(), outNode, edge.outParam, bInputPrim, false, inParamType);
                else
                    updateWildCardParamTypeRecursive(shared_from_this(), inNode, edge.inParam, bInputPrim, true, outParamType);
            }
        }
    }
    else if (outSocketType == zeno::Socket_WildCard) {
        updateWildCardParamTypeRecursive(shared_from_this(), outNode, edge.outParam, bOutputPrim, false, inParamType);
    }
    else if (inSocketType == zeno::Socket_WildCard) {
        updateWildCardParamTypeRecursive(shared_from_this(), inNode, edge.inParam, bInputPrim, true, outParamType);
    }

    inNode->on_link_added_removed(true, edge.inParam, true);
    outNode->on_link_added_removed(false, edge.outParam, true);

    CALLBACK_NOTIFY(addLink, adjustEdge);
    return true;
}

ZENO_API bool Graph::removeLink(const EdgeInfo& edge) {
    CORE_API_BATCH

    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    if (!outNode)
        return false;

    std::shared_ptr<INode> inNode = getNode(edge.inNode);
    if (!inNode)
        return false;

    //pre checking for param.
    bool bExist = false;
    bool bPrimType = outNode->isPrimitiveType(false, edge.outParam, bExist);
    if (!bExist)
        return false;
    bool bPrimType2 = inNode->isPrimitiveType(true, edge.inParam, bExist);
    if (!bExist && bPrimType != bPrimType2)
        return false;

    outNode->removeLink(false, edge);
    inNode->removeLink(true, edge);
    inNode->mark_dirty(true);

    //删除边后，如果有涉及wildCard类型的param，判断是否需要将所属wildCard组reset
    SocketType inSocketType;
    ParamType inParamType;
    inNode->getParamTypeAndSocketType(edge.inParam, bPrimType, true, inParamType, inSocketType);
    resetWildCardParamsType(inSocketType, inNode, edge.inParam, bPrimType, true);
    SocketType outSocketType;
    ParamType outParamType;
    outNode->getParamTypeAndSocketType(edge.outParam, bPrimType, false, outParamType, outSocketType);
    resetWildCardParamsType(outSocketType, outNode, edge.outParam, bPrimType, false);

    if (!bPrimType2) {
        const ParamObject& inParam = inNode->get_input_obj_param(edge.inParam);
        if (inParam.type == gParamType_List || inParam.type == gParamType_Dict) {
            std::vector<EdgeInfo> inParamLinks = inParam.links;
            if (inParamLinks.size() == 1) {
                if (auto node = getNode(inParamLinks[0].outNode)) {
                    ParamObject existOneParam = node->get_output_obj_param(inParamLinks[0].outParam);
                    if (existOneParam.type == inParam.type) {   //只连一条dict/list,重置tokey表示直连
                        updateLink(inParamLinks[0], false, inParamLinks[0].inKey, "");
                    }
                }
            }
        }
    }

    inNode->on_link_added_removed(true, edge.inParam, false);
    outNode->on_link_added_removed(false, edge.outParam, false);

    CALLBACK_NOTIFY(removeLink, edge)
    return true;
}

ZENO_API bool Graph::removeLinks(const std::string nodename, bool bInput, const std::string paramname)
{
    CORE_API_BATCH

    std::shared_ptr<INode> spNode = getNode(nodename);
    std::vector<EdgeInfo> links = spNode->getLinksByParam(bInput, paramname);
    for (auto link : links)
        removeLink(link);

    CALLBACK_NOTIFY(removeLinks, nodename, bInput, paramname)
    return true;
}

ZENO_API bool Graph::updateLink(const EdgeInfo& edge, bool bInput, const std::string oldkey, const std::string newkey)
{
    CORE_API_BATCH

    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    std::shared_ptr<INode> inNode = getNode(edge.inNode);
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

ZENO_API bool Graph::moveUpLinkKey(const EdgeInfo& edge, bool bInput, const std::string keyName)
{
    CORE_API_BATCH
    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    std::shared_ptr<INode> inNode = getNode(edge.inNode);
    if (!inNode)
        return false;
    return moveUpLinkKey(edge, bInput, keyName);
}

}
