#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>
#include <zeno/utils/safe_at.h>
#include <zeno/core/Descriptor.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <iostream>

namespace zeno {

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{}

ZENO_API Graph::Graph() = default;
ZENO_API Graph::~Graph() = default;

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

namespace {
struct GraphApplyException {
    INode *node;
    std::exception_ptr ep;

    GlobalStatus evalStatus() const {
        try {
            std::rethrow_exception(ep);
        } catch (ErrorException const &e) {
            log_error("==> error during {}: {}", node->myname, e.what());
            return {node, e.getError()};
        } catch (std::exception const &e) {
            log_error("==> exception during {}: {}", node->myname, e.what());
            return {node, std::make_shared<StdError>(std::current_exception())};
        } catch (...) {
            log_error("==> unknown exception during {}", node->myname);
            return {node, std::make_shared<StdError>(std::current_exception())};
        }
        return {};
    }
};
}

ZENO_API void Graph::applyNode(std::string const &id) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return;
    }
    ctx->visited.insert(id);
    auto node = safe_at(nodes, id, "node");
    try {
        node->doApply();
    } catch (GraphApplyException const &gae) {
        throw gae;
    } catch (...) {
        throw GraphApplyException{node, std::current_exception()};
    }
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &ids) {
    ctx = std::make_unique<Context>();
    try {
        for (auto const &id: ids) {
            applyNode(id);
        }
    } catch (GraphApplyException const &gae) {
        *session->globalStatus = gae.evalStatus();
    }
    ctx = nullptr;
}

ZENO_API void Graph::applyNodesToExec() {
    log_debug("{} nodes to exec", nodesToExec.size());
    applyNodes(nodesToExec);
}

ZENO_API void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    safe_at(nodes, dn, "node")->inputBounds[ds] = std::pair(sn, ss);
}

ZENO_API void Graph::setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    safe_at(nodes, id, "node")->inputs[par] = val;
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
        setNodeInput(id, parid, objectFromLiterial(val));
    }, val);
}

}
