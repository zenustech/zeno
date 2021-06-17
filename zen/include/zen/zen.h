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
#include <any>


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
#  define ZENDEPRECATED //__attribute__((deprecated))
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


struct Session;

struct IObject {
    ZENDEPRECATED IObject() = default;

    template <class T>
    ZENDEPRECATED static std::unique_ptr<T> make() {
        return std::make_unique<T>();
    }

    template <class T>
    ZENDEPRECATED static T *as() {
        return std::make_unique<T>();
    }
};

struct Context {
    std::set<std::string> visited;
};

struct INode {
public:
    Session *sess = nullptr;
    std::string myname;
    std::map<std::string, std::pair<std::string, std::string>> inputBounds;
    std::map<std::string, std::string> inputs;
    std::map<std::string, std::string> outputs;
    std::map<std::string, IValue> params;

    ZENAPI INode();
    ZENAPI ~INode();

    ZENAPI void doApply();
    ZENAPI virtual void complete();

protected:
    virtual void apply() = 0;

    ZENAPI bool has_input(std::string const &id) const;

    struct my_std_any {
        std::any *_m_p;

        my_std_any(std::any *p) : _m_p(p) {
        }

        template <class T>
        ZENDEPRECATED T *as() { return &std::any_cast<T &>(*_m_p); }

        operator std::any &() { return *_m_p; }
    };

    ZENAPI my_std_any *get_input(std::string const &id) const;

    template <class T>
    T *get_input(std::string const &id) const {
        return &std::any_cast<T &>(*get_input(id));
    }

    ZENAPI std::string get_input_ref(std::string const &id) const;

    ZENAPI IValue get_param(std::string const &id) const;

    template <class T>
    T get_param(std::string const &id) const {
        return std::get<T>(get_param(id));
    }

    ZENAPI std::any *new_output(std::string const &id);

    ZENAPI void set_output_ref(std::string const &id, std::string const &ref);

    /*template <class T>
    T *new_output(std::string const &id) {
        auto obj = new_output(id);
        *obj = std::make_any<T>();
        return &std::any_cast<T &>(*obj);
    }*/

    template <class T>
    ZENDEPRECATED T *new_member(std::string const &id) {
        auto obj = std::make_unique<T>();
        auto obj_ptr = obj.get();
        set_output(id, std::move(obj));
        return obj_ptr;
    }

    template <class T>
    ZENDEPRECATED void set_output(std::string const &id, std::unique_ptr<T> &&obj) {
        *new_output(id) = std::make_any<T>(*obj);
    }

    template <class T>
    ZENDEPRECATED void set_output(std::string const &id, std::unique_ptr<T> &obj) {
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

struct Session {
    std::map<std::string, std::unique_ptr<std::any>> objects;
    std::map<std::string, std::unique_ptr<INode>> nodes;
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::unique_ptr<Context> ctx;

    ZENAPI void _defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls);
    ZENAPI std::string getNodeOutput(std::string const &sn, std::string const &ss) const;
    ZENAPI std::any *getObject(std::string const &id) const;

    template <class F>
    int defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
        _defNodeClass(id, std::make_unique<ImplNodeClass<F>>(ctor, desc));
        return 1;
    }

    ZENAPI void clearNodes();
    ZENAPI void applyNodes(std::vector<std::string> const &ids);
    ZENAPI void addNode(std::string const &cls, std::string const &id);
    ZENAPI void applyNode(std::string const &id);
    ZENAPI void completeNode(std::string const &id);
    ZENAPI void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss);
    ZENAPI std::string dumpDescriptors() const;
    ZENAPI void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val);
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

inline void clearNodes() {
    return getSession().clearNodes();
}

inline void addNode(std::string const &cls, std::string const &id) {
    return getSession().addNode(cls, id);
}

inline void completeNode(std::string const &id) {
    return getSession().completeNode(id);
}

inline void applyNodes(std::vector<std::string> const &ids) {
    return getSession().applyNodes(ids);
}

inline void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    return getSession().bindNodeInput(dn, ds, sn, ss);
}

inline void setNodeParam(std::string const &id, std::string const &par,
        IValue const &val) {
    return getSession().setNodeParam(id, par, val);
}



#define ZENDEFNODE(Class, ...) \
    static int def##Class = zen::defNodeClass(std::make_unique<Class>, #Class, __VA_ARGS__)


}
