#pragma once

#include <zeno/core/Graph.h>
#include <zeno/core/Scene.h>
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>

namespace zeno {

template <class T>
[[deprecated("use ZENO_DEFNODE(T)(...)")]]
inline int defNodeClass(std::string const &id, Descriptor const &desc = {}) {
    getSession().defNodeClass(std::make_unique<T>, id, desc);
    return 1;
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

inline void applyNodes(std::set<std::string> const &ids) {
    return getSession().getDefaultScene().getGraph().applyNodes(ids);
}

inline void bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    return getSession().getDefaultScene().getGraph().bindNodeInput(dn, ds, sn, ss);
}

inline void setNodeInputString(std::string const &id, std::string const &par, std::string const &val) {
    return getSession().getDefaultScene().getGraph().setNodeInputString(id, par, val);
}

inline void setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    return getSession().getDefaultScene().getGraph().setNodeInput(id, par, val);
}

inline void setNodeParam(std::string const &id, std::string const &par,
        std::variant<int, float, std::string> const &val) {
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

template <class F>
auto defNodeClassHelper(F const &func, std::string const &name) {
    return [=] (zeno::Descriptor const &desc) -> int {
        getSession().defNodeClass(func, name, desc);
        return 1;
    };
}

template <class F>
auto defOverloadNodeClassHelper(F const &func, std::string const &name, std::vector<std::string> const &types) {
    return [=] (zeno::Descriptor const &desc) -> int {
        getSession().defOverloadNodeClass(func, name, types, desc);
        return 1;
    };
}

template <class T>
using SharedPtr = std::shared_ptr<T>;

template <class T, class ...Ts>
auto makeShared(Ts &&...ts) {
    return std::make_shared<T>(std::forward<Ts>(ts)...);
}

#define ZENO_DEFNODE(Class) \
    static int def##Class = zeno::defNodeClassHelper(std::make_unique<Class>, #Class)

#define ZENO_DEFOVERLOADNODE(Class, PostFix, ...) \
    static int def##Class##PostFix = zeno::defOverloadNodeClassHelper(std::make_unique<Class##PostFix>, #Class, {__VA_ARGS__})

// [[deprecated("use ZENO_DEFNODE(T)(...)")]]
#define ZENDEFNODE(Class, ...) \
    ZENO_DEFNODE(Class)(__VA_ARGS__)

template <class T, class ...Ts>
[[deprecated("use safe_any_cast<T>() instead")]]
auto smart_any_cast(Ts &&...ts) {
    return safe_any_cast<T>(std::forward<Ts>(ts)...);
}

}
