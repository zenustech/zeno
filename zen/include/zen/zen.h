#pragma once


#include <string>
#include <iostream>///sdfgsdfg/ssdfg/
#include <memory>
#include <vector>
#include <variant>
#include <sstream>
#include <sstream>
#include <array>
#include <map>


namespace zen {

typedef std::array<int, 3> int3;
typedef std::array<float, 3> float3;


class Exception : public std::exception {
private:
  std::string msg;
public:
  Exception(std::string const &msg) : msg(msg) {}

  char const *what() const throw() {
    return msg.c_str();
  }
};


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
T safe_at(std::map<std::string, T> const &m,
    std::string const &key, std::string const &msg) {
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


using IValue = std::variant<std::string, int, float, int3, float3>;

template <class T>
T get_float3(IValue const &v) {
  float3 x = std::get<float3>(v);
  return T(x[0], x[1], x[2]);
}

template <class T>
T get_int3(IValue const &v) {
  int3 x = std::get<int3>(v);
  return T(x[0], x[1], x[2]);
}


struct IObject {
  using Ptr = std::unique_ptr<IObject>;

  template <class T>
  static std::unique_ptr<T> make() {
    return std::make_unique<T>();
  }

  template <class T>
  T *as() {
    return dynamic_cast<T *>(this);
  }

  template <class T>
  const T *as() const {
    return dynamic_cast<const T *>(this);
  }

  virtual ~IObject() = default;
};


struct EmptyObject : IObject {
};


struct INode;
static std::string getNodeName(INode *node);
static void setObject(std::string name, IObject::Ptr object);
static IObject *getObject(std::string name);


struct INode {
  using Ptr = std::unique_ptr<INode>;

  virtual void apply() = 0;

  void on_apply() {
    apply();
    // set dummy output sockets for connection order
    //set_output("order", IObject::make<EmptyObject>());
  }

  void set_param(std::string name, IValue const &value) {
    params[name] = value;
  }

  void set_input(std::string name, IObject *value) {
    inputs[name] = value;
  }

  void set_input(std::string name, IObject::Ptr const &value) {
    set_input(name, value.get());
  }

  std::string get_node_name() {
    return getNodeName(this);
  }

  IObject *get_output(std::string name) {
    auto myname = get_node_name();
    return getObject(myname + "::" + name);
  }

protected:
  IValue get_param(std::string name) {
    return safe_at(params, name, "param");
  }

  IObject *get_input(std::string name) {
    return safe_at(inputs, name, "input");
  }

  bool has_input(std::string name) {
    return inputs.find(name) != inputs.end();
  }

  void set_output(std::string name, IObject::Ptr object) {
    auto myname = get_node_name();
    setObject(myname + "::" + name, std::move(object));
  }

  template <class T>
  void set_output(std::string name, std::unique_ptr<T> &object) {
    set_output(name, std::move(object));
  }

  template <class T>
  T *new_member(std::string name) {
    IObject::Ptr object = IObject::make<T>();
    IObject *rawptr = object.get();
    set_output(name, std::move(object));
    return static_cast<T *>(rawptr);
  }

private:
  std::map<std::string, IValue> params;
  std::map<std::string, IObject *> inputs;
};


struct INodeClass {
  using Ptr = std::unique_ptr<INodeClass>;

  virtual std::unique_ptr<INode> new_instance() = 0;
};

template <class T>
struct NodeClass : INodeClass {
  using Ptr = std::unique_ptr<NodeClass>;

  virtual std::unique_ptr<INode> new_instance() override {
    return std::make_unique<T>();
  }
};


struct ParamDescriptor {
  std::string type, name, defl;
};

struct Descriptor {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  int _init = initialize();

  int initialize() {
    // append dummy sockets for perserving exec orders
    //inputs.push_back("order");
    //outputs.push_back("order");
    return 0;
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

  std::string serialize() const {
    std::string res = "";
    res += "(" + join_str(inputs, ",") + ")";
    res += "(" + join_str(outputs, ",") + ")";
    std::vector<std::string> paramStrs;
    for (auto const &[type, name, defl]: params) {
      paramStrs.push_back(type + ":" + name + ":" + defl);
    }
    res += "(" + join_str(paramStrs, ",") + ")";
    res += "(" + join_str(categories, ",") + ")";
    return res;
  }
};


struct Session {
private:
  std::map<std::string, Descriptor> nodeDescriptors;
  std::map<std::string, INodeClass::Ptr> nodeClasses;
  std::map<std::string, IObject::Ptr> objects;
  std::map<std::string, INode::Ptr> nodes;
  std::map<INode *, std::string> nodesRev;
  std::vector<std::pair<void (*)(), void (*)()>> callbacks;

  struct _PrivCtor {};
  static std::unique_ptr<Session> _instance;

public:
  static Session &get() {
    if (!_instance)
      _instance = std::make_unique<Session>(_PrivCtor{});
    return *_instance.get();
  }

  Session(_PrivCtor) {
  }

  void addNode(std::string type, std::string name) {
    auto node = safe_at(nodeClasses, type, "node class")->new_instance();
    nodesRev[node.get()] = name;
    nodes[name] = std::move(node);
  }

  void setNodeParam(std::string name,
      std::string key, IValue const &value) {
    safe_at(nodes, name, "node")->set_param(key, value);
  }

  void setNodeInput(std::string name,
      std::string key, std::string srcname) {
    safe_at(nodes, name, "node")->set_input(key,
        safe_at(objects, srcname, "object"));
  }

  void applyNode(std::string name) {
    safe_at(nodes, name, "node")->on_apply();
  }

  void setObject(std::string name, IObject::Ptr object) {
    objects[name] = std::move(object);
  }

  IObject *getObject(std::string name) {
    return safe_at(objects, name, "object");
  }

  std::string getNodeName(INode *node) {
    return safe_at(nodesRev, node, "node pointer");
  }

  template <class T> // T <- INode
  int defNodeClass(std::string name, Descriptor const &desc) {
    nodeClasses[name] = std::make_unique<NodeClass<T>>();
    nodeDescriptors[name] = desc;
    return 1;
  }

  int defStartStop(void (*start)(), void (*stop)()) {
    callbacks.push_back(std::make_pair(start, stop));
    return 1;
  }

  void initialize() {
    for (auto const &[start, stop]: callbacks)
      start();
  }

  void finalize() {
    for (auto const &[start, stop]: callbacks)
      stop();
  }

  std::string dumpDescriptors() {
    // dump the node descriptors (for node editor),
    // according to the defNodeClass'es in this DLL.
    std::string res = "";
    for (auto const &[key, desc]: nodeDescriptors) {
      res += key + ":" + desc.serialize() + "\n";
    }
    return res;
  }
};


static void addNode(std::string name, std::string type) {
  return Session::get().addNode(name, type);
}

static void setNodeParam(std::string name,
    std::string key, IValue const &value) {
  return Session::get().setNodeParam(name, key, value);
}

static void setNodeInput(std::string name,
    std::string key, std::string srcname) {
  return Session::get().setNodeInput(name, key, srcname);
}

static void applyNode(std::string name) {
  return Session::get().applyNode(name);
}

static void setObject(std::string name, IObject::Ptr object) {
  return Session::get().setObject(name, std::move(object));
}

static IObject *getObject(std::string name) {
  return Session::get().getObject(name);
}

static std::string getNodeName(INode *node) {
  return Session::get().getNodeName(node);
}

template <class T> // T <- INode
int defNodeClass(std::string name, Descriptor const &desc) {
  return Session::get().defNodeClass<T>(name, desc);
}

static int defStartStop(void (*start)(), void (*stop)()) {
  return Session::get().defStartStop(start, stop);
}

static std::string dumpDescriptors() {
  return Session::get().dumpDescriptors();
}

static void initialize() {
  return Session::get().initialize();
}

static void finalize() {
  return Session::get().finalize();
}



#ifdef ZEN_IMPLEMENTATION
std::unique_ptr<Session> Session::_instance;
#endif

}
