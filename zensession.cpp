#define _ZEN_INDLL
#include <zen/zen.h>

namespace zen {


ZENAPI Exception::Exception(std::string const &msg) noexcept
    : msg(msg) {
}

ZENAPI Exception::~Exception() noexcept = default;

ZENAPI char const *Exception::what() const noexcept {
    return msg.c_str();
}

template <class S, class T>
static std::string join_str(std::vector<T> const &elms, S const &delim) {
  std::stringstream ss;
  auto p = elms.begin(), end = elms.end();
  if (p != end)
    ss << *p++;
  for (; p != end; ++p) {
    ss << delim << *p;
  }
  return ss.str();
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
T safe_at(std::map<std::string, T> const &m, std::string const &key,
          std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw std::string("invalid " + msg + " name: " + key);
  }
  return it->second;
}

template <class T, class S>
T safe_at(std::map<S, T> const &m, S const &key, std::string const &msg) {
  auto it = m.find(key);
  if (it == m.end()) {
    throw std::string("invalid " + msg + "as index");
  }
  return it->second;
}

static std::unique_ptr<zen::Session> sess;

ZENAPI Session &getSession() {
  if (!sess) {
    sess = std::make_unique<zen::Session>();
  }
  return *sess;
}

ZENAPI INode::INode() = default;
ZENAPI INode::~INode() = default;

ZENAPI void INode::init() {}

ZENAPI std::vector<std::string> INode::requirements() {
  std::vector<std::string> ret;
  for (auto const &[key, name] : inputs) {
    ret.push_back(name);
  }
  return ret;
}

ZENAPI void INode::on_init() { init(); }

ZENAPI void INode::set_param(std::string const &name, IValue const &value) {
  params[name] = value;
}

ZENAPI void INode::set_input_ref(std::string const &name,
                                 std::string const &srcname) {
  inputs[name] = srcname;
}

ZENAPI std::string INode::get_node_name() { return getNodeName(this); }

ZENAPI IValue INode::get_param(std::string const &name) {
  return safe_at(params, name, "param");
}

ZENAPI std::string INode::get_input_ref(std::string const &name) {
  return safe_at(inputs, name, "input");
}

ZENAPI IObject *INode::get_input(std::string const &name) {
  auto ref = get_input_ref(name);
  return getObject(ref);
}

ZENAPI bool INode::has_input(std::string const &name) {
  return inputs.find(name) != inputs.end();
}

ZENAPI std::string INode::get_output_ref(std::string const &name) {
  auto myname = get_node_name();
  return myname + "::" + name;
}

ZENAPI IObject *INode::get_output(std::string const &name) {
  auto ref = get_output_ref(name);
  return getObject(ref);
}

ZENAPI void INode::set_output(std::string const &name, IObject::Ptr object) {
  auto ref = get_output_ref(name);
  setObject(ref, std::move(object));
}

ZENAPI void INode::set_output_ref(std::string const &name,
                                  std::string const &srcname) {
  auto ref = get_output_ref(name);
  setReference(ref, srcname);
}

ZENAPI void INode::on_apply() {
  bool ok = true;

  // get dummy boolean to see if this node should be executed
  if (has_input("COND")) {
    auto cond = get_input("COND")->as<BooleanObject>();
    ok = cond->value;
  }

  if (ok)
    apply();

  // set dummy output sockets for connection order
  set_output("DST", IObject::make<BooleanObject>());
}

ZENAPI INodeClass::INodeClass() = default;
ZENAPI INodeClass::~INodeClass() = default;

ZENAPI ParamDescriptor::ParamDescriptor(std::string const &type,
                                        std::string const &name,
                                        std::string const &defl)
    : type(type), name(name), defl(defl) {}

ZENAPI ParamDescriptor::~ParamDescriptor() = default;

ZENAPI Descriptor::Descriptor() = default;

ZENAPI Descriptor::Descriptor(std::vector<std::string> const &inputs,
                              std::vector<std::string> const &outputs,
                              std::vector<ParamDescriptor> const &params,
                              std::vector<std::string> const &categories)
    : inputs(inputs), outputs(outputs), params(params), categories(categories) {
  // append dummy sockets for perserving exec orders
  this->inputs.push_back("SRC");
  this->inputs.push_back("COND");
  this->outputs.push_back("DST");
}

ZENAPI Descriptor::~Descriptor() = default;

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

ZENAPI void Session::addNode(std::string const &type, std::string const &name) {
  if (nodes.find(name) != nodes.end())
    return;
  auto node = safe_at(nodeClasses, type, "node class")->new_instance();
  nodesRev[node.get()] = name;
  nodes[name] = std::move(node);
}
ZENAPI std::vector<std::string>
Session::getNodeRequirements(std::string const &name) {
  return safe_at(nodes, name, "node")->requirements();
}
ZENAPI void Session::setNodeParam(std::string const &name,
                                  std::string const &key, IValue const &value) {
  safe_at(nodes, name, "node")->set_param(key, value);
}
ZENAPI void Session::setNodeInput(std::string const &name,
                                  std::string const &key,
                                  std::string const &srcname) {
  safe_at(nodes, name, "node")->set_input_ref(key, srcname);
}
ZENAPI void Session::initNode(std::string const &name) {
  safe_at(nodes, name, "node")->on_init();
}
ZENAPI void Session::applyNode(std::string const &name) {
  safe_at(nodes, name, "node")->on_apply();
}
ZENAPI void Session::setObject(std::string const &name, IObject::Ptr object) {
  objects[name] = std::move(object);
}
ZENAPI bool Session::hasObject(std::string const &name) {
  auto refname = getReference(name).value_or(name);
  return objects.find(refname) != objects.end();
}
ZENAPI IObject *Session::getObject(std::string const &name) {
  auto refname = getReference(name).value_or(name);
  return safe_at(objects, refname, "object");
}
ZENAPI void Session::setReference(std::string const &name,
                                  std::string const &srcname) {
  auto refname = getReference(srcname).value_or(srcname);
  references[name] = refname;
}
ZENAPI std::optional<std::string>
Session::getReference(std::string const &name) {
  auto it = references.find(name);
  if (it == references.end()) {
    return std::nullopt;
  }
  return it->second;
}
ZENAPI std::string Session::getNodeName(INode *node) {
  return safe_at(nodesRev, node, "node pointer");
}
ZENAPI std::string Session::dumpDescriptors() {
  // dump the node descriptors (for node editor),
  // according to the defNodeClass'es in this DLL.
  std::string res = "";
  for (auto const &[key, desc] : nodeDescriptors) {
    res += key + ":" + desc.serialize() + "\n";
  }
  return res;
}
ZENAPI void Session::doDefNodeClass(std::unique_ptr<INodeClass> cls,
                                    std::string const &name,
                                    Descriptor const &desc) {
  nodeClasses[name] = std::move(cls);
  nodeDescriptors[name] = desc;
}
} // namespace zen
