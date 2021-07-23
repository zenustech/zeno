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
#include <zeno/any.h>


#ifdef _MSC_VER
# ifdef DLL_ZENO
#  define ZENAPI __declspec(dllexport)
# else
#  define ZENAPI __declspec(dllimport)
# endif
#else
# define ZENAPI
#endif


namespace zeno {


class Exception : public std::exception {
private:
  std::string msg;
public:
  ZENAPI Exception(std::string const &msg) noexcept;
  ZENAPI ~Exception() noexcept;
  ZENAPI char const *what() const noexcept;
};


using IValue = std::variant<std::string, int, float>;


struct IObject {
#ifndef ZEN_NOREFDLL
    ZENAPI IObject();
    ZENAPI virtual ~IObject();

    ZENAPI virtual std::shared_ptr<IObject> clone() const;
    ZENAPI virtual bool assign(IObject *other);
    ZENAPI virtual void dumpfile(std::string const &path);
#else
    virtual ~IObject() = default;
    virtual std::shared_ptr<IObject> clone() const { return nullptr; }
    virtual bool assign(IObject *other) { return false; }
    virtual void dumpfile(std::string const &path) {}
#endif

    using Ptr = std::unique_ptr<IObject>;

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
    std::map<std::string, shared_any> inputs;
    std::map<std::string, shared_any> outputs;
    std::map<std::string, IValue> params;
    std::set<std::string> options;

    ZENAPI INode();
    ZENAPI virtual ~INode();

    ZENAPI void doComplete();
    ZENAPI virtual void doApply();

protected:
    ZENAPI bool checkApplyCondition();
    ZENAPI void requireInput(std::string const &ds);
    ZENAPI void coreApply();

    ZENAPI virtual void complete();
    ZENAPI virtual void apply() = 0;

    ZENAPI bool has_option(std::string const &id) const;
    ZENAPI bool has_input(std::string const &id) const;
    ZENAPI IValue get_param(std::string const &id) const;
    ZENAPI shared_any const &_get_input(std::string const &id) const;
    ZENAPI void _set_output(std::string const &id,
        shared_any obj);

    //[[deprecated("use shared_any::make instead of std::make_shared<T>")]]
    ZENAPI void set_output(std::string const &id,
        std::shared_ptr<IObject> &&obj) {
        _set_output(id, shared_any::make<std::shared_ptr<IObject>>(obj));
    }

    //[[deprecated("use shared_any::make instead of std::make_shared<T>")]]
    auto get_input(std::string const &id) const {
        return _get_input(id).get<std::shared_ptr<IObject>>();
    }

    [[deprecated("use set_output")]]
    void set_output_ref(std::string const &id,
        std::shared_ptr<IObject> &&obj) {
        set_output(id, std::move(obj));
    }

    template <class T>
    //[[deprecated("use shared_any::make instead of std::make_shared<T>")]]
    std::shared_ptr<T> get_input(std::string const &id) const {
        return std::dynamic_pointer_cast<T>(get_input(id));
    }

    template <class T>
    T get_param(std::string const &id) const {
        return std::get<T>(get_param(id));
    }


    template <class T>
    [[deprecated("use set_output")]]
    T *new_member(std::string const &id) {
        auto obj = std::make_shared<T>();
        auto obj_ptr = obj.get();
        set_output(id, std::move(obj));
        return obj_ptr;
    }

    template <class T>
    [[deprecated("use set_output(id, std::move(obj))")]]
    void set_output(std::string const &id,
        std::shared_ptr<T> &obj) {
        set_output(id, std::move(obj));
    }
};

struct ParamDescriptor {
  std::string type, name, defl;

  ZENAPI ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl);
  ZENAPI ~ParamDescriptor();
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
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  ZENAPI Descriptor();
  ZENAPI Descriptor(
	  std::vector<std::string> const &inputs,
	  std::vector<std::string> const &outputs,
	  std::vector<ParamDescriptor> const &params,
	  std::vector<std::string> const &categories);

  ZENAPI std::string serialize() const;
};

struct INodeClass {
    std::unique_ptr<Descriptor> desc;

    ZENAPI INodeClass(Descriptor const &desc);
    ZENAPI virtual ~INodeClass();

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

struct Context {
    std::set<std::string> visited;

    inline void mergeVisited(Context const &other) {
        visited.insert(other.visited.begin(), other.visited.end());
    }

    ZENAPI Context();
    ZENAPI Context(Context const &other);
    ZENAPI ~Context();
};

struct Graph {
    Session *sess = nullptr;

    std::map<std::string, std::unique_ptr<INode>> nodes;

    std::map<std::string, shared_any> subInputs;
    std::map<std::string, shared_any> subOutputs;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, shared_any> portals;

    std::unique_ptr<Context> ctx;

    bool isViewed = true;

    ZENAPI Graph();
    ZENAPI ~Graph();

    ZENAPI void clearNodes();
    ZENAPI void applyNodes(std::vector<std::string> const &ids);
    ZENAPI void addNode(std::string const &cls, std::string const &id);
    ZENAPI void applyNode(std::string const &id);
    ZENAPI void completeNode(std::string const &id);
    ZENAPI void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENAPI void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val);
    ZENAPI void setNodeOptions(std::string const &id,
            std::set<std::string> const &opts);
    ZENAPI shared_any const &getNodeOutput(
        std::string const &sn, std::string const &ss) const;
};

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::map<std::string, std::unique_ptr<Graph>> graphs;
    Graph *currGraph;

    ZENAPI Session();
    ZENAPI ~Session();

    ZENAPI Graph &getGraph() const;
    ZENAPI void switchGraph(std::string const &name);
    ZENAPI std::string dumpDescriptors() const;
    ZENAPI void _defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls);

    template <class F>
    int defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
        _defNodeClass(id, std::make_unique<ImplNodeClass<F>>(ctor, desc));
        return 1;
    }
};


ZENAPI Session &getSession();


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
    return getSession().switchGraph(name);
}

inline void clearNodes() {
    return getSession().getGraph().clearNodes();
}

inline void addNode(std::string const &cls, std::string const &id) {
    return getSession().getGraph().addNode(cls, id);
}

inline void completeNode(std::string const &id) {
    return getSession().getGraph().completeNode(id);
}

inline void applyNodes(std::vector<std::string> const &ids) {
    return getSession().getGraph().applyNodes(ids);
}

inline void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    return getSession().getGraph().bindNodeInput(dn, ds, sn, ss);
}

inline void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val) {
    return getSession().getGraph().setNodeParam(id, par, val);
}

inline void setNodeOptions(std::string const &id,
        std::set<std::string> const &opts) {
    return getSession().getGraph().setNodeOptions(id, opts);
}



#define ZENDEFNODE(Class, ...) \
    static int def##Class = zeno::defNodeClass(std::make_unique<Class>, #Class, __VA_ARGS__)


}
