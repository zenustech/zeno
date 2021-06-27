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

# ifdef _ZEN_INDLL
#  define ZENAPI __declspec(dllexport)
# else
#  define ZENAPI __declspec(dllimport)
# endif
# define ZENDEPRECATED __declspec(deprecated)

#else

# define ZENAPI
# ifdef __GNUC__
#  define ZENDEPRECATED __attribute__((deprecated))
# else
#  define ZENDEPRECATED
# endif

#endif


namespace zen {


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
#ifndef _ZEN_FREE_IOBJECT
    ZENAPI IObject();
    ZENAPI virtual ~IObject();

    ZENAPI virtual std::shared_ptr<IObject> clone() const;
    ZENAPI virtual void visualize() const;
#else
    virtual ~IObject() = default;
    virtual std::shared_ptr<IObject> clone() const { return nullptr; }
    ZENAPI virtual void visualize() const {}
#endif

    using Ptr = std::unique_ptr<IObject>;

    template <class T>
    ZENDEPRECATED static std::shared_ptr<T> make() { return std::make_shared<T>(); }

    template <class T>
    ZENDEPRECATED T *as() { return dynamic_cast<T *>(this); }

    template <class T>
    ZENDEPRECATED const T *as() const { return dynamic_cast<const T *>(this); }
};

template <class Derived, class Base = IObject>
struct IObjectClone : Base {
    virtual std::shared_ptr<IObject> clone() const {
        return std::make_shared<Derived>(static_cast<Derived const &>(*this));
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
    std::map<std::string, std::string> inputs;
    std::map<std::string, std::string> outputs;
    std::map<std::string, IValue> params;
    std::set<std::string> options;

    ZENAPI INode();
    ZENAPI ~INode();

    ZENAPI void doComplete();
    ZENAPI virtual void doApply();

protected:
    bool has_executed = false;
    bool has_executed_complete = false;

    ZENAPI bool checkApplyCondition();

    ZENAPI virtual void complete();
    ZENAPI virtual void apply() = 0;

    ZENAPI bool has_option(std::string const &id) const;
    ZENAPI bool has_input(std::string const &id) const;
    ZENAPI std::shared_ptr<IObject> get_input(std::string const &id) const;
    ZENAPI std::string get_input_ref(std::string const &id) const;
    ZENAPI IValue get_param(std::string const &id) const;
    ZENAPI void set_output(std::string const &id,
        std::shared_ptr<IObject> &&obj);
    ZENAPI void set_output_ref(std::string const &id, std::string const &ref);

    template <class T>
    std::shared_ptr<T> get_input(std::string const &id) const {
        return std::dynamic_pointer_cast<T>(get_input(id));
    }

    template <class T>
    T get_param(std::string const &id) const {
        return std::get<T>(get_param(id));
    }


    template <class T>
    ZENDEPRECATED T *new_member(std::string const &id) {
        auto obj = std::make_shared<T>();
        auto obj_ptr = obj.get();
        set_output(id, std::move(obj));
        return obj_ptr;
    }

    template <class T>
    ZENDEPRECATED void set_output(std::string const &id,
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

    INodeClass(Descriptor const &desc)
        : desc(std::make_unique<Descriptor>(desc)) {}

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

    ZENAPI Context();
    ZENAPI Context(Context const &other);
    ZENAPI ~Context();
};

struct Graph {
    Session *sess = nullptr;

    std::map<std::string, std::unique_ptr<INode>> nodes;
    std::map<std::string, std::shared_ptr<IObject>> objects;

    std::map<std::string, std::shared_ptr<IObject>> subInputs;
    std::map<std::string, std::shared_ptr<IObject>> subOutputs;

    std::map<std::string, std::string> portalIns;
    std::map<std::string, std::shared_ptr<IObject>> portals;

    std::unique_ptr<Context> ctx;

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
    ZENAPI std::string getNodeOutput(std::string const &sn, std::string const &ss) const;
    ZENAPI std::shared_ptr<IObject> const &getObject(std::string const &id) const;
};

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::map<std::string, std::unique_ptr<Graph>> graphs;
    std::vector<std::shared_ptr<zen::IObject>> viewObjects;
    Graph *currGraph;

    ZENAPI Session();
    ZENAPI ~Session();

    ZENAPI void endCurrFrame();
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
ZENDEPRECATED inline int defNodeClass(std::string const &id, Descriptor const &desc = {}) {
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

inline void endCurrFrame() {
    return getSession().endCurrFrame();
}



#define ZENDEFNODE(Class, ...) \
    static int def##Class = zen::defNodeClass(std::make_unique<Class>, #Class, __VA_ARGS__)


}
