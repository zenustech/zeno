#pragma once

#include <zeno/utils/defs.h>
#include <zeno/utils/UserData.h>
#include <zeno/core/Descriptor.h>
#include <functional>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct Session;
struct Scene;
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

struct Session {
    using ObjectMethod = std::function<void(UserData const &, std::vector<IObject *>)>;
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;
    std::map<std::string, ObjectMethod> objectMethods;
    std::unique_ptr<Scene> defaultScene;

    ZENO_API Session();
    ZENO_API ~Session();

    ZENO_API Scene &getDefaultScene();
    ZENO_API std::unique_ptr<Scene> createScene();
    ZENO_API std::string dumpDescriptors() const;
    ZENO_API void defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls);
    ZENO_API void callObjectMethod(
            std::string const &name, UserData &ud,
            std::vector<IObject *> const &args);
    ZENO_API void defObjectMethod(
            std::string const &name,
            ObjectMethod const &func,
            std::vector<std::string> const &types);

    template <class F>
    void defNodeClass(F const &ctor, std::string const &id, Descriptor const &desc = {}) {
        defNodeClass(id, std::make_unique<ImplNodeClass<F>>(ctor, desc));
    }
};

ZENO_API Session &getSession();

}
