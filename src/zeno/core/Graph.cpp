#include <zeno/core/Scene.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>
#include <zeno/utils/safe_at.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{}

ZENO_API Graph::Graph() = default;
ZENO_API Graph::~Graph() = default;

ZENO_API void Graph::setGraphInputPromise(std::string const &id,
        std::function<zany()> getter) {
    subInputPromises[id] = std::move(getter);
}

ZENO_API void Graph::setGraphInput(std::string const &id, zany obj) {
    subInputs[id] = std::move(obj);
}

ZENO_API void Graph::applyGraph() {
    applyNodes(finalOutputNodes);
}

ZENO_API zany const &Graph::getGraphOutput(
        std::string const &id) const {
    return safe_at(subOutputs, id, "subgraph output");
}

ZENO_API zany const &Graph::getNodeOutput(
    std::string const &sn, std::string const &ss) const {
    auto node = safe_at(nodes, sn, "node");
    if (node->muted_output)
        return node->muted_output;
    return safe_at(node->outputs, ss, "output", node->myname);
}

ZENO_API void Graph::clearNodes() {
    nodes.clear();
}

ZENO_API void Graph::addNode(std::string const &cls, std::string const &id) {
    if (nodes.find(id) != nodes.end())
        return;  // no add twice, to prevent output object invalid
    auto cl = safe_at(session->nodeClasses, cls, "node class");
    auto node = cl->new_instance();
    node->graph = this;
    node->myname = id;
    node->nodeClass = cl;
    nodes[id] = std::move(node);
}

ZENO_API void Graph::completeNode(std::string const &id) {
    safe_at(nodes, id, "node")->doComplete();
}

ZENO_API void Graph::applyNode(std::string const &id) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return;
    }
    ctx->visited.insert(id);
    auto node = safe_at(nodes, id, "node");
    try {
        node->doApply();
    } catch (std::exception const &e) {
        throw zeno::BaseException("During evaluation of `"
                + node->myname + "`:\n" + e.what());
    }
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &ids) {
    try {
        ctx = std::make_unique<Context>();
        for (auto const &id: ids) {
            applyNode(id);
        }
        ctx = nullptr;
    } catch (std::exception const &e) {
        ctx = nullptr;
        throw zeno::BaseException(
                (std::string)"ZENO Traceback (most recent call last):\n"
                + e.what());
    }
}

ZENO_API void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    safe_at(nodes, dn, "node")->inputBounds[ds] = std::pair(sn, ss);
}

ZENO_API void Graph::setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    safe_at(nodes, id, "node")->inputs[par] = val;
}

ZENO_API void Graph::setNodeOption(std::string const &id,
        std::string const &name) {
    safe_at(nodes, id, "node")->options.insert(name);
}

ZENO_API std::set<std::string> Graph::getGraphInputNames() const {
    std::set<std::string> res;
    for (auto const &[id, _]: subInputNodes) {
        res.insert(id);
    }
    return res;
}

ZENO_API std::set<std::string> Graph::getGraphOutputNames() const {
    std::set<std::string> res;
    for (auto const &[id, _]: subOutputNodes) {
        res.insert(id);
    }
    return res;
}

ZENO_API UserData &Graph::getUserData() {
    return userData;
}

ZENO_API std::unique_ptr<INode> Graph::getOverloadNode(std::string const &id,
        std::vector<std::shared_ptr<IObject>> const &inputs) const {
    auto node = session->getOverloadNode(id, inputs);
    if (!node) return nullptr;
    node->graph = const_cast<Graph *>(this);
    return node;
}

ZENO_API void Graph::setNodeParam(std::string const &id, std::string const &par,
    std::variant<int, float, std::string> const &val) {
    auto parid = par + ":";
    std::visit([&] (auto const &val) {
        if constexpr (std::is_same_v<std::decay_t<decltype(val)>, std::string>) {
            setNodeInput(id, parid, std::make_shared<StringObject>(val));
        } else {
            setNodeInput(id, parid, std::make_shared<NumericObject>(val));
        }
    }, val);
}

}
