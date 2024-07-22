#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
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
#include <zeno/core/ReferManager.h>
#include <zeno/core/GlobalVariable.h>
#include "reflect/reflection.generated.hpp"


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

ZENO_API bool Graph::applyNode(std::string const &node_name) {
    const std::string uuid = safe_at(m_name2uuid, node_name, "uuid");
    auto node = safe_at(m_nodes, uuid, "node name").get();
    GraphException::translated([&] {
        node->doApply();
    }, node);
    return true;
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &nodes) {
    ctx = std::make_unique<Context>();

    scope_exit _{[&] {
        ctx = nullptr;
    }};

    for (auto const& node_name: nodes) {
        applyNode(node_name);
    }
}

ZENO_API void Graph::runGraph() {
    log_debug("{} nodes to exec", m_viewnodes.size());
    applyNodes(m_viewnodes);
}

void Graph::onNodeParamUpdated(PrimitiveParam* spParam, zeno::reflect::Any old_value, zeno::reflect::Any new_value) {
    assert(spParam);
    if (Param_String == spParam->type) {
        auto spNode = spParam->m_wpNode.lock();
        assert(spNode);

        const std::string& nodecls = spNode->get_nodecls();
        const std::string& uuid = spNode->get_uuid();

        std::string newstr = zeno_get<std::string>(old_value);

        frame_nodes.erase(uuid);
        getSession().globalVariableManager->removeDependGlobalVaraible(spNode->get_uuid_path(), "$F");

        std::regex pattern("\\$F");
        if (std::regex_search(newstr, pattern, std::regex_constants::match_default)) {
            frame_nodes.insert(uuid);
            getSession().globalVariableManager->addDependGlobalVaraible(spNode->get_uuid_path(), "$F", zeno::reflect::type_info<int>());
        }
    }
    else if (Param_Vec2f == spParam->type) {
        auto spNode = spParam->m_wpNode.lock();
        assert(spNode);
        const std::string& uuid = spNode->get_uuid();
        frame_nodes.erase(uuid);
        getSession().globalVariableManager->removeDependGlobalVaraible(spNode->get_uuid_path(), "$F");
        {
            vec2s vec;
            if (zeno_get_if(new_value, vec)) {
                std::regex pattern("\\$F");
                for (auto val : vec) {
                    if (std::regex_search(val, pattern)) {
                        frame_nodes.insert(uuid);
                        getSession().globalVariableManager->addDependGlobalVaraible(spNode->get_uuid_path(), "$F", zeno::reflect::type_info<int>());
                    }
                }
            }
        }
    }
    else if (Param_Vec3f == spParam->type) {
        auto spNode = spParam->m_wpNode.lock();
        assert(spNode);
        const std::string& uuid = spNode->get_uuid();
        frame_nodes.erase(uuid);
        getSession().globalVariableManager->removeDependGlobalVaraible(spNode->get_uuid_path(), "$F");
        {
            vec3s vec;
            if (zeno_get_if(new_value, vec)) {
                std::regex pattern("\\$F");
                for (auto val : vec) {
                    if (std::regex_search(val, pattern)) {
                        frame_nodes.insert(uuid);
                        getSession().globalVariableManager->addDependGlobalVaraible(spNode->get_uuid_path(), "$F", zeno::reflect::type_info<int>());
                    }
                }
            }
        }
    }
    else if (Param_Vec4f == spParam->type) {
        auto spNode = spParam->m_wpNode.lock();
        assert(spNode);
        const std::string& uuid = spNode->get_uuid();
        frame_nodes.erase(uuid);
        getSession().globalVariableManager->removeDependGlobalVaraible(spNode->get_uuid_path(), "$F");
        {
            vec4s vec;
            if (zeno_get_if(new_value, vec)) {
                std::regex pattern("\\$F");
                for (auto val : vec) {
                    if (std::regex_search(val, pattern)) {
                        frame_nodes.insert(uuid);
                        getSession().globalVariableManager->addDependGlobalVaraible(spNode->get_uuid_path(), "$F", zeno::reflect::type_info<int>());
                    }
                }
            }
        }
    }
}

void Graph::viewNodeUpdated(const std::string node, bool bView) {
    if (bView) {
        //TODO: only run calculation chain which associate with `node`.
        //getSession().run_main_graph();
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
        else if (node.cls == "Subnet")
        {
            if (std::shared_ptr<zeno::SubnetNode> sbn = std::dynamic_pointer_cast<zeno::SubnetNode>(spNode))
                sbn->setCustomUi(node.customUi);
        }
    }
    //import edges
    for (const auto& link : graph.links) {
        if (!isLinkVaild(link))
            continue;
        std::shared_ptr<INode> outNode = getNode(link.outNode);
        std::shared_ptr<INode> inNode = getNode(link.inNode);

        bool bExist = false;
        bool bOutputPrim = outNode->isPrimitiveType(false, link.outParam, bExist);
        bool bInputPrim = inNode->isPrimitiveType(true, link.inParam, bExist);

        if (bInputPrim) {
            std::shared_ptr<PrimitiveLink> spLink = std::make_shared<PrimitiveLink>();
            outNode->init_primitive_link(false, link.outParam, spLink);
            inNode->init_primitive_link(true, link.inParam, spLink);
        }
        else {
            std::shared_ptr<ObjectLink> spLink = std::make_shared<ObjectLink>();
            spLink->fromkey = link.outKey;
            spLink->tokey = link.inKey;
            outNode->init_object_link(false, link.outParam, spLink);
            inNode->init_object_link(true, link.inParam, spLink);
        }
    }
}

void Graph::markDirtyWhenFrameChanged()
{
    for (const std::string& uuid : frame_nodes) {
        m_nodes[uuid]->mark_dirty(true);
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
    std::string oldPath = zeno::objPathToStr(spNode->get_path());
    std::string name = newName;
    if (m_name2uuid.find(name) != m_name2uuid.end()) {
        name = generateNewName(spNode->get_nodecls());
    }
    spNode->set_name(name);

    m_name2uuid[name] = m_name2uuid[oldName];
    m_name2uuid.erase(oldName);

    sync_to_set(m_viewnodes, oldName, name);

    //sync_to_set(frame_nodes, oldName, newName);
    //sync_to_set(subnet_nodes, oldName, newName);
    //sync_to_set(asset_nodes, oldName, newName);
    //sync_to_set(subinput_nodes, oldName, newName);
    //sync_to_set(suboutput_nodes, oldName, newName);

    CALLBACK_NOTIFY(updateNodeName, oldName, name)
    //update refer
    const auto& referMgr = getSession().referManager;
    std::string newPath = zeno::objPathToStr(spNode->get_path());
    std::string uuid_path = zeno::objPathToStr(spNode->get_uuid_path());
    referMgr->updateReferParam(oldPath, newPath, uuid_path);
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
    if (cls == "Subnet") {
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

ZENO_API std::shared_ptr<INode> Graph::getNodeByPath(std::string path)
{
    if (path.empty())
        return nullptr;
    int sPos = 0;
    if (m_name == "main")
    {
        sPos = path.find(m_name) + m_name.size();
        if (path.size() <= sPos)
            return nullptr;
        path = path.substr(sPos + 1, path.size() - sPos);
    }
    std::string name = path.substr(0, path.find("/"));
    if (m_name2uuid.find(name) == m_name2uuid.end())
        return nullptr;
    std::string uuid = m_name2uuid[name];
    auto it = m_nodes.find(uuid);
    if (it == m_nodes.end()) {
        return nullptr;
    }
    sPos = path.find("/");

    if (sPos != std::string::npos)
    {
        path = path.substr(sPos + 1, path.size() - sPos);
        //subnet
        if (std::shared_ptr<SubnetNode> subnetNode = std::dynamic_pointer_cast<SubnetNode>(it->second))
        {
            auto spGraph = subnetNode->subgraph;
            if (spGraph)
                return spGraph->getNodeByPath(path);
            else
                return nullptr;
        }
    }
    return it->second;
}

ZENO_API std::map<std::string, std::shared_ptr<INode>> Graph::getNodes() const {
    return m_nodes;
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

    node_set[nodecls].erase(name);
    m_nodes.erase(uuid);

    frame_nodes.erase(uuid);
    subnet_nodes.erase(uuid);
    asset_nodes.erase(uuid);
    m_viewnodes.erase(name);
    m_name2uuid.erase(name);
    //remove refer info
    const auto& referMgr = getSession().referManager;
    std::string path = zeno::objPathToStr(spNode->get_path());
    auto objPath = zeno::objPathToStr(spNode->get_uuid_path());
    referMgr->removeReference(path, objPath);
    CALLBACK_NOTIFY(removeNode, name)
    return true;
}

bool zeno::Graph::isLinkVaild(const EdgeInfo& edge)
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
    if (bOutputPrim)
    {
        const auto& spParam = outNode->get_output_prim_param(edge.outParam);
        outSocketType = spParam.socketType;
        outParamType = spParam.type;
    }
    else
    {
        const auto& spParam = outNode->get_output_obj_param(edge.outParam);
        outSocketType = spParam.socketType;
        outParamType = spParam.type;
    }
    SocketType inSocketType;
    ParamType inParamType;
    if (bOutputPrim)
    {
        const auto& spParam = inNode->get_input_prim_param(edge.inParam);
        inSocketType = spParam.socketType;
        inParamType = spParam.type;
    }
    else
    {
        const auto& spParam = inNode->get_input_obj_param(edge.inParam);
        inSocketType = spParam.socketType;
        inParamType = spParam.type;
    }

    if (outSocketType == zeno::Socket_WildCard && inSocketType == zeno::Socket_WildCard)
    {
        zeno::log_warn("wildcard can not link wildcard.");
        return false;
    }
    if (outSocketType == zeno::Socket_WildCard)
    {
        const auto& params = outNode->getWildCardParams(edge.outParam, bOutputPrim);
        for (const auto& param : params)
        {
            outNode->update_param_type(param, bOutputPrim, inParamType);
            if (param == edge.outParam)
                outParamType = inParamType;
        }
    }
    if (inSocketType == zeno::Socket_WildCard)
    {
        const auto& params = inNode->getWildCardParams(edge.inParam, bInputPrim);
        for (const auto& param : params)
        {
            inNode->update_param_type(param, bInputPrim, outParamType);
            if (param == edge.inParam)
                inParamType = outParamType;
        }
        if (inNode->get_nodecls() == "SubOutput" && bInputPrim) { //当一个节点连接到SubOutput时，更新parentSubnet对应param的类型
            if (std::shared_ptr<Graph> graph = inNode->getGraph().lock()) {
                if (graph->optParentSubgNode.has_value()) {
                    if (SubnetNode* parentSubgNode = optParentSubgNode.value()) {
                        parentSubgNode->update_param_type(inNode->get_name(), true, outParamType);
                    }
                }
            }
        }
    }
    if (inParamType != outParamType)
    {
        zeno::log_warn("param type no match.");
        return false;
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

    if (!isLinkVaild(edge))
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
        if (inParam.type == Param_Dict || inParam.type == Param_List) {
            bool bSameType = inParam.type == outParam.type;
            if (bSameType) {
                //直接连接，并去掉输入端原来的参数.
                if (edge.inKey.empty()) {
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
        outNode->init_primitive_link(false, edge.outParam, spLink);
        inNode->init_primitive_link(true, edge.inParam, spLink);
        adjustEdge.bObjLink = false;
    }
    else {
        std::shared_ptr<ObjectLink> spLink = std::make_shared<ObjectLink>();
        spLink->fromkey = edge.outKey;
        spLink->tokey = edge.inKey;
        outNode->init_object_link(false, edge.outParam, spLink);
        inNode->init_object_link(true, edge.inParam, spLink);
        adjustEdge.bObjLink = true;
    }

    inNode->mark_dirty(true);

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
    return inNode->updateLinkKey(true, edge.inParam, oldkey, newkey);
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
