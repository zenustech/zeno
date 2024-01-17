#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/core/Descriptor.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GraphException.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <zeno/core/IParam.h>
#include <zeno/utils/uuid.h>
#include <iostream>

namespace zeno {

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{}

ZENO_API Graph::Graph(const std::string& name) : name(name) {
}

ZENO_API Graph::~Graph() {

}

ZENO_API zany Graph::getNodeInput(std::string const& sn, std::string const& ss) const {
    auto node = safe_at(nodes, sn, "node name").get();
    return node->get_input(ss);
}

ZENO_API void Graph::clearNodes() {
    nodes.clear();
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

ZENO_API Graph *Graph::getSubnetGraph(std::string const &id) const {
    auto node = static_cast<SubnetNode *>(safe_at(nodes, id, "node name").get());
    return node->subgraph.get();
}

ZENO_API void Graph::completeNode(std::string const &id) {
    safe_at(nodes, id, "node name")->doComplete();
}

ZENO_API bool Graph::applyNode(std::string const &id) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return false;
    }
    ctx->visited.insert(id);
    auto node = safe_at(nodes, id, "node name").get();
    GraphException::translated([&] {
        node->doApply();
    }, node->name);
    if (dirtyChecker && dirtyChecker->amIDirty(id)) {
        return true;
    }
    return false;
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &ids) {
    ctx = std::make_unique<Context>();

    scope_exit _{[&] {
        ctx = nullptr;
    }};

    for (auto const &id: ids) {
        applyNode(id);
    }
}

ZENO_API void Graph::applyNodesToExec() {
    log_debug("{} nodes to exec", nodesToExec.size());
    applyNodes(nodesToExec);
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
        std::map<std::string, zany> inputs) const {

    auto cl = safe_at(session->nodeClasses, id, "node class name").get();
    const std::string& name = generateUUID();
    auto se = cl->new_instance(name);
    se->graph = const_cast<Graph*>(this);
    se->directly_setinputs(inputs);
    se->doOnlyApply();
    return se->getoutputs();
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
    name = graph.name;
    //import nodes first.
    for (const auto& [name, node] : graph.nodes) {
        if (node.subgraph) {
            std::shared_ptr<INode> spNode = createSubnetNode(node.cls);
            std::shared_ptr<SubnetNode> subnetNode = std::dynamic_pointer_cast<SubnetNode>(spNode);
            assert(subnetNode);
            subnetNode->init(node);
        }
        else {
            std::shared_ptr<INode> spNode = createNode(node.cls);
            spNode->init(node);
            if (node.cls == "SubInput") {
                std::string name;
                for (const ParamInfo& param : node.inputs) {
                    if (param.name == "name") {
                        //todo: exception.
                        name = std::get<std::string>(param.defl);
                        break;
                    }
                }
                subInputNodes[name] = node.name;
            }
            else if (node.cls == "SubOutput") {
                std::string name;
                for (const ParamInfo& param : node.inputs) {
                    if (param.name == "name") {
                        //todo: exception.
                        name = std::get<std::string>(param.defl);
                        break;
                    }
                }
                subOutputNodes[name] = node.name;
            }
        }
    }
    //import edges
    for (const auto& link : graph.links) {
        std::shared_ptr<INode> outNode = getNode(link.outNode);
        std::shared_ptr<INode> inNode = getNode(link.inNode);
        assert(outNode && inNode);

        std::shared_ptr<IParam> outParam = outNode->get_output_param(link.outParam);
        if (!outParam) {
            zeno::log_warn("no output param `{}` on node `{}`", link.outParam, link.outNode);
            continue;
        }

        std::shared_ptr<IParam> inParam = inNode->get_input_param(link.inParam);
        if (!inParam) {
            zeno::log_warn("no input param `{}` on node `{}`", link.inParam, link.inNode);
            continue;
        }

        std::shared_ptr<ILink> spLink = std::make_shared<zeno::ILink>();
        spLink->fromparam = outParam;
        spLink->toparam = inParam;
        spLink->keyName = link.inKey;
        outParam->links.emplace_back(spLink);
        inParam->links.emplace_back(spLink);
    }
}

std::string Graph::generateNewName(const std::string& node_cls)
{
    if (node_set.find(node_cls) == node_set.end())
        node_set.insert(std::make_pair(node_cls, std::set<std::string>()));

    auto& nodes = node_set[node_cls];
    int i = 1;
    while (true) {
        std::string new_name = node_cls + std::to_string(1);
        if (nodes.find(new_name) == nodes.end()) {
            nodes.insert(new_name);
            return new_name;
        }
    }
    return "";
}

ZENO_API std::shared_ptr<INode> Graph::createNode(std::string const& cls)
{
    auto cl = safe_at(session->nodeClasses, cls, "node class name").get();
    std::string const& name = generateNewName(cls);
    auto node = cl->new_instance(name);
    node->graph = this;
    node->nodeClass = cl;
    nodes[name] = node;

    CALLBACK_NOTIFY(createNode, name, node)
    return node;
}

ZENO_API std::shared_ptr<INode> Graph::createSubnetNode(std::string const& cls)
{
    auto subcl = std::make_unique<ImplSubnetNodeClass>();
    std::string const& name = generateNewName(cls);
    auto node = subcl->new_instance(name);
    node->graph = this;
    node->nodeClass = subcl.get();

    auto subnetnode = std::dynamic_pointer_cast<SubnetNode>(node);
    subnetnode->subnetClass = std::move(subcl);

    nodes[node->name] = node;
    CALLBACK_NOTIFY(createSubnetNode, subnetnode)
    return node;
}

ZENO_API Graph* Graph::addSubnetNode(std::string const& id) {
    //deprecated:
    return nullptr;
#if 0
    auto subcl = std::make_unique<ImplSubnetNodeClass>();
    auto node = subcl->new_instance(id);
    node->graph = this;
    node->name = id;
    node->nodeClass = subcl.get();
    auto subnode = static_cast<SubnetNode*>(node.get());
    subnode->subgraph->session = this->session;
    subnode->subnetClass = std::move(subcl);
    auto subg = subnode->subgraph.get();
    nodes[id] = std::move(node);
    return subg;
#endif
}

std::map<std::string, std::string> Graph::getSubInputs()
{
    return subInputNodes;
}

std::map<std::string, std::string> Graph::getSubOutputs()
{
    return subOutputNodes;
}

ZENO_API std::shared_ptr<INode> Graph::getNode(std::string const& name) {
    if (nodes.find(name) == nodes.end())
        return nullptr;
    return nodes[name];
}

ZENO_API std::string Graph::getName() const {
    return name;
}

ZENO_API bool Graph::removeNode(std::string const& name) {
    auto it = nodes.find(name);
    if (it == nodes.end())
        return false;

    node_set[it->second->nodecls].erase(name);
    nodes.erase(name);
    CALLBACK_NOTIFY(removeNode, name)
    return true;
}

ZENO_API bool Graph::addLink(const EdgeInfo& edge) {
    std::shared_ptr<INode> outNode = getNode(edge.outNode);
    if (!outNode)
        return false;
    std::shared_ptr<INode> inNode = getNode(edge.inNode);
    if (!inNode)
        return false;

    std::shared_ptr<IParam> outParam = outNode->get_output_param(edge.outParam);
    std::shared_ptr<IParam> inParam = inNode->get_input_param(edge.inParam);

    std::shared_ptr<ILink> spLink = std::make_shared<ILink>();
    outParam->links.push_back(spLink);
    inParam->links.push_back(spLink);

    CALLBACK_NOTIFY(addLink, edge);
    return true;
}

ZENO_API bool Graph::removeLink(const EdgeInfo& edge) {
    //TODO: DaMi implement this.
    return false;
}

}
