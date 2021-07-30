#pragma once


#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <optional>
#include <sstream>
#include <array>
#include <map>
#include <set>


#ifdef _MSC_VER
# ifdef DLL_ZENO
#  define ZENO_API __declspec(dllexport)
# else
#  define ZENO_API __declspec(dllimport)
# endif
#else
# define ZENO_API
#endif


namespace zeno {


class Exception : public std::exception {
private:
  std::string msg;
public:
  ZENO_API Exception(std::string const &msg) noexcept;
  ZENO_API ~Exception() noexcept;
  ZENO_API char const *what() const noexcept;
};


using IValue = std::variant<std::string, int, float>;


struct IObject {
#ifndef ZENO_APIFREE
    ZENO_API IObject();
    ZENO_API virtual ~IObject();

    ZENO_API virtual std::shared_ptr<IObject> clone() const;
    ZENO_API virtual bool assign(IObject *other);
    ZENO_API virtual void dumpfile(std::string const &path);
#else
    virtual ~IObject() = default;
    virtual std::shared_ptr<IObject> clone() const { return nullptr; }
    virtual bool assign(IObject *other) { return false; }
    virtual void dumpfile(std::string const &path) {}
#endif

    template <class T>
    [[deprecated("use std::make_shared<T>")]]
    static std::shared_ptr<T> make() { return std::make_shared<T>(); }

    template <class T>
    [[deprecated("use dynamic_cast<T *>")]]
    T *as() { return dynamic_cast<T *>(this); }

    template <class T>
    [[deprecated("use dynamic_cast<const T *>")]]
    const T *as() const { return dynamic_cast<const T *>(this); }
};

template <class Derived, class Base = IObject>
struct IObjectClone : Base {
    virtual std::shared_ptr<IObject> clone() const {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
    }

    virtual bool assign(IObject *other) {
        auto src = dynamic_cast<Derived *>(other);
        if (!src)
            return false;
        auto dst = static_cast<Derived *>(this);
        *dst = *src;
        return true;
    }
};

struct Graph;
struct INodeClass;

struct INode {
public:
    Graph *graph = nullptr;
    INodeClass *nodeClass = nullptr;

    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, std::shared_ptr<IObject>> inputs;
    std::map<std::string, std::shared_ptr<IObject>> outputs;
    std::shared_ptr<IObject> muted_output;
    std::map<std::string, IValue> params;
    std::set<std::string> options;

    ZENO_API INode();
    ZENO_API virtual ~INode();

    ZENO_API void doComplete();
    ZENO_API virtual void doApply();

protected:
    ZENO_API bool checkApplyCondition();
    ZENO_API void requireInput(std::string const &ds);
    ZENO_API void coreApply();

    ZENO_API virtual void complete();
    ZENO_API virtual void apply() = 0;

    ZENO_API bool has_option(std::string const &id) const;
    ZENO_API bool has_input(std::string const &id) const;
    ZENO_API IValue get_param(std::string const &id) const;
    ZENO_API std::shared_ptr<IObject> get_input(std::string const &id) const;
    ZENO_API void set_output(std::string const &id,
        std::shared_ptr<IObject> &&obj);

    template <class T>
    bool has_input(std::string const &id) const {
        if (!has_input(id))
            return false;
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        return (bool)p;
    }

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        auto obj = get_input(id);
        auto p = std::dynamic_pointer_cast<T>(std::move(obj));
        if (!p) {
            throw Exception("input socket `" + id + "` expect `"
                    + typeid(T).name() + "`, got `"
                    + typeid(*obj.get()).name() + "`");
        }
        return p;
    }

    template <class T>
    T get_param(std::string const &id) const {
        return std::get<T>(get_param(id));
    }
};

struct ParamDescriptor {
  std::string type, name, defl;

  ZENO_API ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl);
  ZENO_API ~ParamDescriptor();
};

struct SocketDescriptor {
  std::string type, name, defl;

  ZENO_API SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl = {});
  ZENO_API ~SocketDescriptor();

  //[[deprecated("use {\"sockType\", \"sockName\"} instead of \"sockName\"")]]
  SocketDescriptor(const char *name)
      : SocketDescriptor({}, name) {}
};

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

struct Descriptor {
  std::vector<SocketDescriptor> inputs;
  std::vector<SocketDescriptor> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  ZENO_API Descriptor();
  ZENO_API Descriptor(
	  std::vector<SocketDescriptor> const &inputs,
	  std::vector<SocketDescriptor> const &outputs,
	  std::vector<ParamDescriptor> const &params,
	  std::vector<std::string> const &categories);

  ZENO_API std::string serialize() const;
};

struct INodeClass {
    std::unique_ptr<Descriptor> desc;

    ZENO_API INodeClass(Descriptor const &desc);
    ZENO_API virtual ~INodeClass();

    virtual std::unique_ptr<INode> new_instance() const = 0;
};

template <class F>
struct ImplNodeClass : INodeClass {
    F const &ctor;

    ImplNodeClass(F const &ctor, Descriptor const &desc)
        : INodeClass(desc), ctor(ctor) {}

    virtual std::unique_ptr<INode> new_instance() const override {
        return ctor();
    }
};

struct Session;
struct Scene;

struct Context {
    std::set<std::string> visited;

    inline void mergeVisited(Context const &other) {
        visited.insert(other.visited.begin(), other.visited.end());
    }

    ZENO_API Context();
    ZENO_API Context(Context const &other);
    ZENO_API ~Context();
};

struct Graph {
    Scene *scene = nullptr;

    std::map<std::string, std::unique_ptr<INode>> nodes;

    std::map<std::string, std::shared_ptr<IObject>> subInputs;
    std::map<std::string, std::shared_ptr<IObject>> subOutputs;
    std::map<std::string, std::string> subOutputNodes;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, std::shared_ptr<IObject>> portals;

    std::unique_ptr<Context> ctx;

    bool isViewed = true;
    bool hasAnyView = false;

    ZENO_API Graph();
    ZENO_API ~Graph();

    ZENO_API void setGraphInput(std::string const &id,
            std::shared_ptr<IObject> obj);
    ZENO_API std::shared_ptr<IObject> getGraphOutput(
            std::string const &id) const;
    ZENO_API void applyGraph();

    template <class T>
    std::shared_ptr<T> getGraphOutput(
            std::string const &id) const {
        auto obj = getGraphOutput(id);
        auto p = std::dynamic_pointer_cast<T>(obj);
        if (!p) {
            throw Exception("graph output `" + id + "` expect `"
                    + typeid(T).name() + "`, got `"
                    + typeid(*obj.get()).name() + "`");
        }
        return p;
    }

    ZENO_API void clearNodes();
    ZENO_API void applyNodes(std::vector<std::string> const &ids);
    ZENO_API void addNode(std::string const &cls, std::string const &id);
    ZENO_API void applyNode(std::string const &id);
    ZENO_API void completeNode(std::string const &id);
    ZENO_API void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENO_API void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val);
    ZENO_API void setNodeOption(std::string const &id, std::string const &name);
    ZENO_API std::shared_ptr<IObject> const &getNodeOutput(
        std::string const &sn, std::string const &ss) const;
};

struct Scene {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::map<std::string, std::unique_ptr<Graph>> graphs;

    Graph *currGraph = nullptr;
    Session *sess = nullptr;

    ZENO_API Scene();
    ZENO_API ~Scene();

    ZENO_API Graph &getGraph();
    ZENO_API void clearAllState();
    ZENO_API Graph &getGraph(std::string const &name) const;
    ZENO_API void switchGraph(std::string const &name);
    ZENO_API void loadScene(const char *json);
};

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::unique_ptr<Scene> defaultScene;

    ZENO_API Session();
    ZENO_API ~Session();

    ZENO_API Scene &getDefaultScene();
    ZENO_API std::unique_ptr<Scene> createScene();
    ZENO_API std::string dumpDescriptors() const;
    ZENO_API void _defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls);

    template <class F>
    int defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
        _defNodeClass(id, std::make_unique<ImplNodeClass<F>>(ctor, desc));
        return 1;
    }
};


ZENO_API Session &getSession();


template <class F>
inline int defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
    return getSession().defNodeClass(ctor, id, desc);
}

template <class T>
[[deprecated("use ZENDEFNODE(T, ...)")]]
inline int defNodeClass(std::string const &id, Descriptor const &desc = {}) {
    return getSession().defNodeClass(std::make_unique<T>, id, desc);
}

inline std::string dumpDescriptors() {
    return getSession().dumpDescriptors();
}

inline void switchGraph(std::string const &name) {
    return getSession().getDefaultScene().switchGraph(name);
}

inline void clearAllState() {
    return getSession().getDefaultScene().clearAllState();
}

inline void clearNodes() {
    return getSession().getDefaultScene().getGraph().clearNodes();
}

inline void addNode(std::string const &cls, std::string const &id) {
    return getSession().getDefaultScene().getGraph().addNode(cls, id);
}

inline void completeNode(std::string const &id) {
    return getSession().getDefaultScene().getGraph().completeNode(id);
}

inline void applyNodes(std::vector<std::string> const &ids) {
    return getSession().getDefaultScene().getGraph().applyNodes(ids);
}

inline void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    return getSession().getDefaultScene().getGraph().bindNodeInput(dn, ds, sn, ss);
}

inline void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val) {
    return getSession().getDefaultScene().getGraph().setNodeParam(id, par, val);
}

inline void setNodeOption(std::string const &id, std::string const &name) {
    return getSession().getDefaultScene().getGraph().setNodeOption(id, name);
}

inline void loadScene(const char *json) {
    return getSession().getDefaultScene().loadScene(json);
}

inline std::unique_ptr<Scene> createScene() {
    return getSession().createScene();
}



#define ZENDEFNODE(Class, ...) \
    static int def##Class = zeno::defNodeClass(std::make_unique<Class>, #Class, __VA_ARGS__)


}
