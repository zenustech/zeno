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


#ifdef _MSC_VER
#ifdef _ZEN_INDLL
#define ZENAPI __declspec(dllexport)
#else
#define ZENAPI __declspec(dllimport)
#endif
#else
#define ZENAPI
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

  ZENAPI INode();
  ZENAPI ~INode();

  virtual void apply() = 0;
  ZENAPI virtual void init();
  ZENAPI virtual std::vector<std::string> requirements();
  ZENAPI void on_init();
  ZENAPI void on_apply();
  ZENAPI void set_param(std::string const &name, IValue const &value);
  ZENAPI void set_input_ref(std::string const &name, std::string const &srcname);
  ZENAPI std::string get_node_name();

protected:
	ZENAPI IValue get_param(std::string const &name);
	ZENAPI std::string get_input_ref(std::string const &name);
	ZENAPI IObject *get_input(std::string const &name);
	ZENAPI bool has_input(std::string const &name);
	ZENAPI std::string get_output_ref(std::string const &name);
	ZENAPI IObject *get_output(std::string const &name);
	ZENAPI void set_output(std::string const &name, IObject::Ptr object);
	ZENAPI void set_output_ref(std::string const &name, std::string const &srcname);

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

  ZENAPI INodeClass();
  ZENAPI ~INodeClass();
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

  ZENAPI ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl);
  ZENAPI ~ParamDescriptor();
};

struct Descriptor {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  ZENAPI Descriptor();
  ZENAPI Descriptor(
	  std::vector<std::string> inputs,
	  std::vector<std::string> outputs,
	  std::vector<ParamDescriptor> params,
	  std::vector<std::string> categories);
  ZENAPI ~Descriptor();

  ZENAPI std::string serialize() const;
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
  ZENAPI void addNode(std::string const &type, std::string const &name);

  ZENAPI std::vector<std::string> getNodeRequirements(std::string const &name);

  ZENAPI void setNodeParam(std::string const &name,
	  std::string const &key, IValue const &value);

  ZENAPI void setNodeInput(std::string const &name,
	  std::string const &key, std::string const &srcname);

  ZENAPI void initNode(std::string const &name);

  ZENAPI void applyNode(std::string const &name);

  ZENAPI void setObject(std::string const &name, IObject::Ptr object);

  ZENAPI bool hasObject(std::string const &name);

  ZENAPI IObject *getObject(std::string const &name);

  ZENAPI void setReference(std::string const &name, std::string const &srcname);

  ZENAPI std::optional<std::string> getReference(std::string const &name);

  ZENAPI std::string getNodeName(INode *node);

  ZENAPI std::string dumpDescriptors();

  ZENAPI void doDefNodeClass(std::unique_ptr<INodeClass> cls,
	  std::string const &name, Descriptor const &desc);

  template <class T> // T <- INode
  int defNodeClass(std::string const &name, Descriptor const &desc) {
    return defNodeClassByCtor(std::make_unique<T>, name, desc);
  }

  template <class T> // T <- std::unique_ptr<INode>()
  int defNodeClassByCtor(T const &ctor,
      std::string const &name, Descriptor const &desc) {
    doDefNodeClass(std::make_unique<NodeClass<T>>(ctor), name, desc);
    return 1;
  }
};


extern ZENAPI Session &getSession();

inline void addNode(std::string const &name, std::string const &type) {
  return getSession().addNode(name, type);
}

inline void setNodeParam(std::string const &name,
    std::string const &key, IValue const &value) {
  return getSession().setNodeParam(name, key, value);
}

inline void setNodeInput(std::string const &name,
    std::string const &key, std::string const &srcname) {
  return getSession().setNodeInput(name, key, srcname);
}

inline void initNode(std::string const &name) {
  return getSession().initNode(name);
}

inline void applyNode(std::string const &name) {
  return getSession().applyNode(name);
}

inline void setObject(std::string const &name, IObject::Ptr object) {
  return getSession().setObject(name, std::move(object));
}

inline bool hasObject(std::string const &name) {
  return getSession().hasObject(name);
}

inline IObject *getObject(std::string const &name) {
  return getSession().getObject(name);
}

inline void setReference(std::string const &name, std::string const &srcname) {
  return getSession().setReference(name, srcname);
}

inline std::optional<std::string> getReference(std::string const &name) {
  return getSession().getReference(name);
}

inline std::string getNodeName(INode *node) {
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

inline std::string dumpDescriptors() {
  return getSession().dumpDescriptors();
}

inline std::vector<std::string> getNodeRequirements(std::string name) {
  return getSession().getNodeRequirements(name);
}


}





#if 0
#include <cstdio>
#include <cassert>
#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <Windows.h>
#endif


namespace zen {


static Session &getSession() {

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
            const char path[] = "zensession.dll";

            hdll = ::LoadLibraryExA(path, NULL, NULL);
            if (!hdll) {
                printf("failed to open %s: %d\n", path, ::GetLastError());
                abort();
            }
            proc = (void *)::GetProcAddress(hdll, symbol);
            if (!proc) {
                printf("failed to load symbol %s: %d\n", symbol, ::GetLastError());
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

    static std::unique_ptr<DLLSession> dll;
    if (!dll) {
        dll = std::make_unique<DLLSession>();
    }
    return *dll->getSession();
}


}
#endif
