#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/Descriptor.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct Graph;
struct Session;
struct INode;

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

struct IObject;
struct GlobalState;
struct GlobalComm;
struct GlobalStatus;
struct Translator;
struct Initializer;

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;

    std::unique_ptr<GlobalState> const globalState;
    std::unique_ptr<GlobalComm> const globalComm;
    std::unique_ptr<GlobalStatus> const globalStatus;
    std::unique_ptr<Translator> const translator;
    std::unique_ptr<Initializer> const initializer;

    ZENO_API Session();
    ZENO_API ~Session();

    Session(Session const &) = delete;
    Session &operator=(Session const &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    ZENO_API std::unique_ptr<Graph> createGraph();
    ZENO_API std::string dumpDescriptors() const;
    ZENO_API void defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls);
    ZENO_API void defOverloadNodeClass(std::string const &id, std::vector<std::string> const &types,
            std::unique_ptr<INodeClass> &&cls);
    ZENO_API std::unique_ptr<INode> getOverloadNode(
            std::string const &name, std::vector<std::shared_ptr<IObject>> const &inputs);

    template <class F>
    void defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
        defNodeClass(id, std::make_unique<ImplNodeClass<F>>(ctor, desc));
    }

    template <class F>
    void defOverloadNodeClass(F const &ctor, std::string const &id,
            std::vector<std::string> const &types, Descriptor const &desc = {}) {
        defOverloadNodeClass(id, types, std::make_unique<ImplNodeClass<F>>(ctor, desc));
    }
};

ZENO_API Session &getSession();

}
