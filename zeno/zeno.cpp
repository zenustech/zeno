#include <zeno/zeno.h>
#include <zeno/ConditionObject.h>
#ifdef ZENO_VISUALIZATION  // TODO: can we decouple then from zeno core?
#include <zeno/Visualization.h>
#endif
#ifdef ZENO_GLOBALSTATE
#include <zeno/GlobalState.h>
#endif
#include <zeno/safe_at.h>
#include <cassert>

namespace zeno {

ZENO_API Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
}

ZENO_API Exception::~Exception() noexcept = default;

ZENO_API char const *Exception::what() const noexcept {
    return msg.c_str();
}

ZENO_API IObject::IObject() = default;
ZENO_API IObject::~IObject() = default;

ZENO_API std::shared_ptr<IObject> IObject::clone() const {
    return nullptr;
}

ZENO_API bool IObject::assign(IObject *other) {
    return false;
}

ZENO_API void IObject::dumpfile(std::string const &path) {
}

ZENO_API INode::INode() = default;
ZENO_API INode::~INode() = default;

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{
}

ZENO_API Graph::Graph() = default;
ZENO_API Graph::~Graph() = default;

ZENO_API void Graph::setGraphInput(std::string const &id,
        std::shared_ptr<IObject> &&obj) {
    subInputs[id] = obj;
}

ZENO_API void Graph::applyGraph() {
    std::vector<std::string> applies;
    for (auto const &[id, nodename]: subOutputNodes) {
        applies.push_back(nodename);
    }
    applyNodes(applies);
}

ZENO_API std::shared_ptr<IObject> Graph::getGraphOutput(
        std::string const &id) const {
    return subOutputs.at(id);
}

ZENO_API void INode::doComplete() {
    set_output("DST", std::make_shared<ConditionObject>());
    complete();
}

ZENO_API void INode::complete() {}

ZENO_API bool INode::checkApplyCondition() {
    /*if (has_input("COND")) {  // deprecated
        auto cond = get_input<zeno::ConditionObject>("COND");
        if (!cond->get())
            return false;
    }*/

#ifdef ZENO_GLOBALSTATE
    if (has_option("ONCE")) {
        if (!zeno::state.isFirstSubstep())
            return false;
    }

    if (has_option("PREP")) {
        if (!zeno::state.isOneSubstep())
            return false;
    }
#endif

    if (has_option("MUTE")) {
        auto desc = nodeClass->desc.get();
        if (desc->inputs[0].name != "SRC") {
            muted_output = get_input(desc->inputs[0].name);
        } else {
            for (auto const &[ds, bound]: inputBounds) {
                muted_output = get_input(ds);
                break;
            }
        }
        return false;
    }

    return true;
}

ZENO_API void INode::doApply() {
    for (auto const &[ds, bound]: inputBounds) {
        requireInput(ds);
    }

    coreApply();
}

ZENO_API void INode::requireInput(std::string const &ds) {
    auto [sn, ss] = inputBounds.at(ds);
    graph->applyNode(sn);
    auto ref = graph->getNodeOutput(sn, ss);
    inputs[ds] = ref;
}

ZENO_API void INode::coreApply() {
    if (checkApplyCondition()) {
        apply();
    }

#ifdef ZENO_VISUALIZATION
    if (has_option("VIEW")) {
        graph->hasAnyView = true;
        if (!state.isOneSubstep())  // no duplicate view when multi-substep used
            return;
        if (!graph->isViewed)  // VIEW subnodes only if subgraph is VIEW'ed
            return;
        auto desc = nodeClass->desc.get();
        auto obj = muted_output ? muted_output
            : safe_at(outputs, desc->outputs[0].name, "output");
        auto path = Visualization::exportPath();
        obj->dumpfile(path);
    }
#endif
}

ZENO_API bool INode::has_option(std::string const &id) const {
    return options.find(id) != options.end();
}

ZENO_API bool INode::has_input(std::string const &id) const {
    return inputBounds.find(id) != inputBounds.end();
}

ZENO_API std::shared_ptr<IObject> INode::get_input(std::string const &id) const {
    return safe_at(inputs, id, "input", myname);
}

ZENO_API IValue INode::get_param(std::string const &id) const {
    return safe_at(params, id, "param", myname);
}

ZENO_API void INode::set_output(std::string const &id, std::shared_ptr<IObject> &&obj) {
    outputs[id] = std::move(obj);
}

ZENO_API std::shared_ptr<IObject> const &Graph::getNodeOutput(
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
    auto cl = safe_at(sess->nodeClasses, cls, "node class");
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
        throw zeno::Exception("During evaluation of `"
                + node->myname + "`:\n" + e.what());
    }
}

ZENO_API void Graph::applyNodes(std::vector<std::string> const &ids) {
    try {
        ctx = std::make_unique<Context>();
        for (auto const &id: ids) {
            applyNode(id);
        }
        ctx = nullptr;
    } catch (std::exception const &e) {
        ctx = nullptr;
        throw zeno::Exception(
                (std::string)"ZENO Traceback (most recent call last):\n"
                + e.what());
    }
}

ZENO_API void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    safe_at(nodes, dn, "node")->inputBounds[ds] = std::pair(sn, ss);
}

ZENO_API void Graph::setNodeParam(std::string const &id, std::string const &par,
        IValue const &val) {
    safe_at(nodes, id, "node")->params[par] = val;
}

ZENO_API void Graph::setNodeOption(std::string const &id,
        std::string const &name) {
    safe_at(nodes, id, "node")->options.insert(name);
}


ZENO_API Session::Session() {
    switchGraph("main");
}

ZENO_API Session::~Session() = default;

ZENO_API void Session::clearAllState() {
    graphs.clear();
}

ZENO_API void Session::_defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls) {
    nodeClasses[id] = std::move(cls);
}

ZENO_API INodeClass::INodeClass(Descriptor const &desc)
        : desc(std::make_unique<Descriptor>(desc)) {
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API void Session::switchGraph(std::string const &name) {
    if (graphs.find(name) == graphs.end()) {
        auto subg = std::make_unique<zeno::Graph>();
        subg->sess = this;
        graphs[name] = std::move(subg);
    }
    currGraph = graphs.at(name).get();
}

ZENO_API Graph &Session::getGraph() const {
    return *currGraph;
}

ZENO_API Graph &Session::getGraph(std::string const &name) const {
    return *graphs.at(name);
}

ZENO_API std::string Session::dumpDescriptors() const {
  std::string res = "";
  for (auto const &[key, cls] : nodeClasses) {
    res += "DESC@" + key + "@" + cls->desc->serialize() + "\n";
  }
  return res;
}


ZENO_API Session &getSession() {
    static std::unique_ptr<Session> ptr;
    if (!ptr) {
        ptr = std::make_unique<Session>();
    }
    return *ptr;
}


SocketDescriptor::SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
SocketDescriptor::~SocketDescriptor() = default;


ParamDescriptor::ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
ParamDescriptor::~ParamDescriptor() = default;

ZENO_API Descriptor::Descriptor() = default;
ZENO_API Descriptor::Descriptor(
  std::vector<SocketDescriptor> const &inputs,
  std::vector<SocketDescriptor> const &outputs,
  std::vector<ParamDescriptor> const &params,
  std::vector<std::string> const &categories)
  : inputs(inputs), outputs(outputs), params(params), categories(categories) {
    this->inputs.push_back("SRC");
    //this->inputs.push_back("COND");  // deprecated
    this->outputs.push_back("DST");
}

ZENO_API std::string Descriptor::serialize() const {
  std::string res = "";
  std::vector<std::string> strs;
  for (auto const &[type, name, defl] : inputs) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  strs.clear();
  for (auto const &[type, name, defl] : outputs) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  strs.clear();
  for (auto const &[type, name, defl] : params) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  res += "{" + join_str(categories, "%") + "}";
  return res;
}


} // namespace zeno
