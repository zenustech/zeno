#pragma once

#include "Descriptor.h"

struct Session;
struct Scene;

namespace zeno {

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

}
