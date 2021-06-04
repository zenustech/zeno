#pragma once


#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <optional>
#include <sstream>
#include <sstream>
#include <array>
#include <map>


namespace zen {


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


using IValue = std::variant<std::string, int, float>;


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


struct BooleanObject : IObject {
  bool value{true};
};


struct INode;
static std::string getNodeName(INode *node);
static void setObject(std::string const &name, IObject::Ptr object);
static void setReference(std::string const &name, std::string const &srcname);
static IObject *getObject(std::string const &name);


struct INode {
  using Ptr = std::unique_ptr<INode>;

  virtual void apply() = 0;

  virtual void init() {
  }

  virtual std::vector<std::string> requirements() {
    std::vector<std::string> ret;
    for (auto const &[key, name]: inputs) {
      ret.push_back(name);
    }
    return ret;
  }

  void on_init() {
    init();
  }

  void on_apply() {
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

  void set_param(std::string const &name, IValue const &value) {
    params[name] = value;
  }

  void set_input_ref(std::string const &name, std::string const &srcname) {
    inputs[name] = srcname;
  }

  std::string get_node_name() {
    return getNodeName(this);
  }

protected:
  IValue get_param(std::string const &name) {
    return safe_at(params, name, "param");
  }

  std::string get_input_ref(std::string const &name) {
    return safe_at(inputs, name, "input");
  }

  IObject *get_input(std::string const &name) {
    auto ref = get_input_ref(name);
    return getObject(ref);
  }

  bool has_input(std::string const &name) {
    return inputs.find(name) != inputs.end();
  }

  std::string get_output_ref(std::string const &name) {
    auto myname = get_node_name();
    return myname + "::" + name;
  }

  IObject *get_output(std::string const &name) {
    auto ref = get_output_ref(name);
    return getObject(ref);
  }

  void set_output(std::string const &name, IObject::Ptr object) {
    auto ref = get_output_ref(name);
    setObject(ref, std::move(object));
  }

  void set_output_ref(std::string const &name, std::string const &srcname) {
    auto ref = get_output_ref(name);
    setReference(ref, srcname);
  }

  template <class T>
  void set_output(std::string const &name, std::unique_ptr<T> &object) {
    set_output(name, std::move(object));
  }

  template <class T>
  T *new_member(std::string const &name) {
    IObject::Ptr object = IObject::make<T>();
    IObject *rawptr = object.get();
    set_output(name, std::move(object));
    return static_cast<T *>(rawptr);
  }

private:
  std::map<std::string, IValue> params;
  std::map<std::string, std::string> inputs;
};


struct INodeClass {
  using Ptr = std::unique_ptr<INodeClass>;

  virtual std::unique_ptr<INode> new_instance() = 0;
};


template <class T>
struct NodeClass : INodeClass {
  T const &ctor;

  NodeClass(T const &ctor)
    : ctor(ctor)
  {}

  virtual std::unique_ptr<INode> new_instance() override {
    return ctor();
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
    inputs.push_back("SRC");
    inputs.push_back("COND");
    outputs.push_back("DST");
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
  std::map<std::string, std::string> references;
  std::map<std::string, INode::Ptr> nodes;
  std::map<INode *, std::string> nodesRev;

public:
  void addNode(std::string const &type, std::string const &name) {
    if (nodes.find(name) != nodes.end())
      return;
    auto node = safe_at(nodeClasses, type, "node class")->new_instance();
    nodesRev[node.get()] = name;
    nodes[name] = std::move(node);
  }

  std::vector<std::string> getNodeRequirements(std::string const &name) {
    return safe_at(nodes, name, "node")->requirements();
  }

  void setNodeParam(std::string const &name,
      std::string const &key, IValue const &value) {
    safe_at(nodes, name, "node")->set_param(key, value);
  }

  void setNodeInput(std::string const &name,
      std::string const &key, std::string const &srcname) {
    safe_at(nodes, name, "node")->set_input_ref(key, srcname);
  }

  void initNode(std::string const &name) {
    safe_at(nodes, name, "node")->on_init();
  }

  void applyNode(std::string const &name) {
    safe_at(nodes, name, "node")->on_apply();
  }

  void setObject(std::string const &name, IObject::Ptr object) {
    objects[name] = std::move(object);
  }

  bool hasObject(std::string const &name) {
    auto refname = getReference(name).value_or(name);
    return objects.find(refname) != objects.end();
  }

  IObject *getObject(std::string const &name) {
    auto refname = getReference(name).value_or(name);
    return safe_at(objects, refname, "object");
  }

  void setReference(std::string const &name, std::string const &srcname) {
    auto refname = getReference(srcname).value_or(srcname);
    references[name] = refname;
  }

  std::optional<std::string> getReference(std::string const &name) {
    auto it = references.find(name);
    if (it == references.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::string getNodeName(INode *node) {
    return safe_at(nodesRev, node, "node pointer");
  }

  template <class T> // T <- INode
  int defNodeClass(std::string const &name, Descriptor const &desc) {
    return defNodeClassByCtor(std::make_unique<T>, name, desc);
  }

  template <class T> // T <- std::unique_ptr<INode>()
  int defNodeClassByCtor(T const &ctor,
      std::string name, Descriptor const &desc) {
    nodeClasses[name] = std::make_unique<NodeClass<T>>(ctor);
    nodeDescriptors[name] = desc;
    return 1;
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


static Session &getSession();

static void addNode(std::string const &name, std::string const &type) {
  return getSession().addNode(name, type);
}

static void setNodeParam(std::string const &name,
    std::string const &key, IValue const &value) {
  return getSession().setNodeParam(name, key, value);
}

static void setNodeInput(std::string const &name,
    std::string const &key, std::string const &srcname) {
  return getSession().setNodeInput(name, key, srcname);
}

static void initNode(std::string const &name) {
  return getSession().initNode(name);
}

static void applyNode(std::string const &name) {
  return getSession().applyNode(name);
}

static void setObject(std::string const &name, IObject::Ptr object) {
  return getSession().setObject(name, std::move(object));
}

static bool hasObject(std::string const &name) {
  return getSession().hasObject(name);
}

static IObject *getObject(std::string const &name) {
  return getSession().getObject(name);
}

static void setReference(std::string const &name, std::string const &srcname) {
  return getSession().setReference(name, srcname);
}

static std::optional<std::string> getReference(std::string const &name) {
  return getSession().getReference(name);
}

static std::string getNodeName(INode *node) {
  return getSession().getNodeName(node);
}

template <class T> // T <- INode
int defNodeClass(std::string name, Descriptor const &desc) {
  return getSession().defNodeClass<T>(name, desc);
}

template <class T> // T <- std::unique_ptr<INode>()
int defNodeClassByCtor(T const &ctor,
    std::string name, Descriptor const &desc) {
  return getSession().defNodeClassByCtor<T>(ctor, name, desc);
}

static std::string dumpDescriptors() {
  return getSession().dumpDescriptors();
}

static std::vector<std::string> getNodeRequirements(std::string name) {
  return getSession().getNodeRequirements(name);
}


}





#include <cstdio>
#include <cassert>
#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <Windows.h>
#endif


namespace zen {


struct DLLSession {
    void *proc;

#if defined(__linux__)
    void *hdll;

    DLLSession() {
        const char symbol[] = "__zensession_getSession_v1";
        const char path[] = "libzensession.so";

        hdll = ::dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (!hdll) {
            char const *err = dlerror();
            printf("failed to open %s: %s\n", path, err ? err : "no error");
            abort();
        }
        void *proc = ::dlsym(hdll, symbol);
        if (!proc) {
            char const *err = dlerror();
            printf("failed to load symbol %s: %s\n", symbol, err ? err : "no error");
            abort();
        }
    }

    ~DLLSession() {
        ::dlclose(hdll);
        hdll = nullptr;
    }
#elif defined(_WIN32)
    ::HINSTANCE hdll;

    DLLSession() {
        const char symbol[] = "__zensession_getSession_v1";
        const char path[] = "libzensession.dll";

        hdll = ::LoadLibraryExA(path, NULL, NULL);
        if (!hdll) {
            printf("failed to open %s: %s\n", path, GetLastError());
            abort();
        }
        proc = (void *)::GetProcAddress(hdll, symbol);
        if (!proc) {
            printf("failed to open %s: %s\n", symbol, GetLastError());
            abort();
        }
    }

    ~DLLSession() {
        ::FreeLibrary(hdll);
    }
#else
#error "only windows and linux are supported for now"
#endif

    Session *getSession() {
        return ((Session *(*)())proc)();
    }
};


static Session &getSession() {
    static Session *sess = nullptr;
    if (!sess) {
#if defined(__linux__)
        void *hdll = ::dlopen("libzensession.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if (!hdll) {
            char const *err = dlerror();
            printf("failed to open libzensession.so: %s\n", err ? err : "no error");
            abort();
        }
        void *proc = ::dlsym(hdll, "__zensession_getSession_v1");
        if (!proc) {
            char const *err = dlerror();
            printf("failed to load symbol: %s\n", err ? err : "no error");
            abort();
        }
        sess = ((Session *(*)())proc)();
        ::dlclose(hdll);
        hdll = nullptr;
#elif defined(_WIN32)
#else
        auto hdll = ::LoadLibraryExA("libzensession.dll", NULL, NULL);
        if (!hdll) {
            printf("failed to open libzensession.dll: %s\n", GetLastError());
            abort();
        }
        void *proc = dlsym(hdll, "__zensession_getSession_v1");
        assert(proc);
        sess = ((Session *(*)())proc)();
        assert(sess);
        dlclose(hdll);
        hdll = nullptr;
#error "only windows and linux are supported for now"
#endif
        assert(sess);
    }
    return *sess;
}


}
