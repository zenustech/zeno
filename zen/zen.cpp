#include <zen/zen.h>
#include <zen/ConditionObject.h>
#include <cassert>

namespace zen {

ZENAPI Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
}

ZENAPI Exception::~Exception() noexcept = default;

ZENAPI char const *Exception::what() const noexcept {
    return msg.c_str();
}

template <class T>
T *safe_at(std::map<std::string, std::unique_ptr<T>> const &m,
           std::string const &key, std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw Exception("invalid " + msg + " name: " + key);
  }
  return it->second.get();
}

template <class T>
T const &safe_at(std::map<std::string, T> const &m, std::string const &key,
          std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw Exception("invalid " + msg + " name: " + key);
  }
  return it->second;
}

template <class T, class S>
T const &safe_at(std::map<S, T> const &m, S const &key, std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw Exception("invalid " + msg + " as index");
  }
  return it->second;
}


#ifndef ZEN_FREE_IOBJECT
ZENAPI IObject::IObject() = default;
ZENAPI IObject::~IObject() = default;

ZENAPI std::shared_ptr<IObject> IObject::clone() const {
    return nullptr;
}
#endif

ZENAPI ArrayObject::ArrayObject() = default;
ZENAPI ArrayObject::~ArrayObject() = default;

ZENAPI bool ArrayObject::isScalar() const {
    return !m_isList;
}

ZENAPI size_t ArrayObject::arraySize() const {
    return m_arr.size();
}

ZENAPI std::optional<size_t> ArrayObject::broadcast(std::optional<size_t> n) const {
    if (isScalar()) {
        return n;
    } else if (n.has_value()) {
        return std::min(n.value(), arraySize());
    } else {
        return arraySize();
    }
}

ZENAPI std::shared_ptr<IObject> const &ArrayObject::at(size_t i) const {
    return m_arr[i % m_arr.size()];
}

ZENAPI void ArrayObject::set(size_t i, std::shared_ptr<IObject> &&obj) {
    if (m_arr.size() < i + 1)
        m_arr.resize(i + 1);
    m_arr[i] = std::move(obj);
}

ZENAPI INode::INode() = default;
ZENAPI INode::~INode() = default;

ZENAPI void Session::refObject(
    std::string const &id) {
    int n = ++objectRefs[id];
    //printf("RO %s %d\n", id.c_str(), n);
}
ZENAPI void Session::refSocket(
    std::string const &sn, std::string const &ss) {
    int n = ++socketRefs[sn + "::" + ss];
    //printf("RS %s::%s %d\n", sn.c_str(), ss.c_str(), n);
}
ZENAPI void Session::derefObject(
    std::string const &id) {
    int n = --objectRefs[id];
    //printf("DO %s %d\n", id.c_str(), n);
}
ZENAPI void Session::derefSocket(
    std::string const &sn, std::string const &ss) {
    int n = --socketRefs[sn + "::" + ss];
    //printf("DS %s::%s %d\n", sn.c_str(), ss.c_str(), n);
}
ZENAPI void Session::gcObject(
    std::string const &sn, std::string const &ss,
    std::string const &id) {
    auto sno = nodes.at(sn).get();
    if (sno->inputs.find("COND") != sno->inputs.end()) {
        // TODO: tmpwalkarnd
        return;
    }
    int n = objectRefs[id];
    int m = socketRefs[sn + "::" + ss];
    //printf("GC %s::%s/%s %d/%d\n", sn.c_str(), ss.c_str(), id.c_str(), m, n);
    if (n <= 0 && m <= 0) {
        assert(!n && !m);
        objectRefs.erase(id);
        socketRefs.erase(sn + "::" + ss);
        objects.erase(id);
    }
}

ZENAPI void INode::doComplete() {
    for (auto [ds, bound]: inputBounds) {
        auto [sn, ss] = bound;
        sess->refSocket(sn, ss);
    }
    complete();
}

ZENAPI void INode::complete() {}

ZENAPI void INode::doApply() {
    std::optional<size_t> siz;

    for (auto const &[ds, bound]: inputBounds) {
        auto [sn, ss] = bound;
        sess->applyNode(sn);
        auto ref = sess->getNodeOutput(sn, ss);
        inputs[ds] = ref;
    }

    bool ok = true;
    if (has_input("COND")) {
        auto cond = get_input<zen::ConditionObject>("COND");
        if (!cond->get())
            ok = false;
    }

    if (ok) {
        m_isList = siz.has_value();
        m_listSize = siz.value_or(1);
        m_listIdx = 0;
        listapply();
    }

    for (auto const &[ds, bound]: inputBounds) {
        auto [sn, ss] = bound;
        sess->derefSocket(sn, ss);
        auto ref = inputs.at(ds);
        sess->derefObject(ref);
        sess->gcObject(sn, ss, ref);
    }

    for (auto const &[id, _]: outputs) {
        auto ref = outputs.at(id);
        sess->gcObject(myname, id, ref);
    }

    m_isList = false;
    m_listIdx = 0;
    set_output("DST", std::make_unique<zen::ConditionObject>());
}

ZENAPI void INode::listapply() {
    for (size_t i = 0; i < m_listSize; i++) {
        m_listIdx = i;
        apply();
    }
}

ZENAPI void INode::apply() {}

ZENAPI bool INode::has_input(std::string const &id) const {
    return inputs.find(id) != inputs.end();
}

ZENAPI std::shared_ptr<IObject> INode::get_input(std::string const &id) const {
    auto ref = safe_at(inputs, id, "input");
    return sess->getObject(ref);
}

ZENAPI IValue INode::get_param(std::string const &id) const {
    return safe_at(params, id, "param");
}

ZENAPI void INode::set_output(std::string const &id, std::shared_ptr<IObject> &&obj) {
    auto objid = myname + "::" + id;
    auto &objlist = sess->objects[objid];
    outputs[id] = objid;
}

ZENAPI void INode::set_output_ref(const std::string &id, const std::string &ref) {
    sess->refObject(ref);
    outputs[id] = ref;
}

ZENAPI std::string INode::get_input_ref(const std::string &id) const {
    return safe_at(inputs, id, "input");
}

ZENAPI void Session::_defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls) {
    nodeClasses[id] = std::move(cls);
}

ZENAPI std::string Session::getNodeOutput(std::string const &sn, std::string const &ss) const {
    auto node = safe_at(nodes, sn, "node");
    return safe_at(node->outputs, ss, "node output");
}

ZENAPI std::shared_ptr<IObject> const &Session::getObject(std::string const &id) const {
    return safe_at(objects, id, "object");
}

ZENAPI void Session::clearNodes() {
    nodes.clear();
    objects.clear();
}

ZENAPI void Session::addNode(std::string const &cls, std::string const &id) {
    if (nodes.find(id) != nodes.end())
        return;  // no add twice, to prevent output object invalid
    auto node = safe_at(nodeClasses, cls, "node class")->new_instance();
    node->sess = this;
    node->myname = id;
    nodes[id] = std::move(node);
}

ZENAPI void Session::completeNode(std::string const &id) {
    safe_at(nodes, id, "node")->doComplete();
}

ZENAPI void Session::applyNode(std::string const &id) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return;
    }
    ctx->visited.insert(id);
    safe_at(nodes, id, "node")->doApply();
}

ZENAPI void Session::applyNodes(std::vector<std::string> const &ids) {
    ctx = std::make_unique<Context>();
    for (auto const &id: ids) {
        applyNode(id);
    }
    ctx = nullptr;
}

ZENAPI void Session::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    safe_at(nodes, dn, "node")->inputBounds[ds] = std::pair(sn, ss);
}

ZENAPI void Session::setNodeParam(std::string const &id, std::string const &par,
        IValue const &val) {
    safe_at(nodes, id, "node")->params[par] = val;
}

ZENAPI std::string Session::dumpDescriptors() const {
  std::string res = "";
  for (auto const &[key, cls] : nodeClasses) {
    res += "DESC:" + key + ":" + cls->desc->serialize() + "\n";
  }
  return res;
}


ZENAPI Session &getSession() {
    static std::unique_ptr<Session> ptr;
    if (!ptr) {
        ptr = std::make_unique<Session>();
    }
    return *ptr;
}



ParamDescriptor::ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
ParamDescriptor::~ParamDescriptor() = default;

ZENAPI Descriptor::Descriptor() = default;
ZENAPI Descriptor::Descriptor(
  std::vector<std::string> const &inputs,
  std::vector<std::string> const &outputs,
  std::vector<ParamDescriptor> const &params,
  std::vector<std::string> const &categories)
  : inputs(inputs), outputs(outputs), params(params), categories(categories) {
    this->inputs.push_back("SRC");
    this->inputs.push_back("COND");
    this->outputs.push_back("DST");
}

ZENAPI std::string Descriptor::serialize() const {
  std::string res = "";
  res += "(" + join_str(inputs, ",") + ")";
  res += "(" + join_str(outputs, ",") + ")";
  std::vector<std::string> paramStrs;
  for (auto const &[type, name, defl] : params) {
      paramStrs.push_back(type + ":" + name + ":" + defl);
  }
  res += "(" + join_str(paramStrs, ",") + ")";
  res += "(" + join_str(categories, ",") + ")";
  return res;
}


} // namespace zen
