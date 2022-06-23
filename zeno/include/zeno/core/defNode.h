#pragma once

#include <zeno/core/Session.h>

namespace zeno {

template <class F>
auto _defNodeClassHelper(F const &func, std::string const &name) {
    return [=] (zeno::Descriptor const &desc) -> int {
        getSession().defNodeClass(func, name, desc);
        return 1;
    };
}

//template <class F>
//auto _defOverloadNodeClassHelper(F const &func, std::string const &name, std::vector<std::string> const &types) {
    //return [=] (zeno::Descriptor const &desc) -> int {
        //getSession().defOverloadNodeClass(func, name, types, desc);
        //return 1;
    //};
//}

#define ZENO_DEFNODE(Class) \
    static int _def##Class = ::zeno::_defNodeClassHelper(std::make_unique<Class>, #Class)

//#define ZENO_DEFOVERLOADNODE(Class, PostFix, ...) \
    //static int _def##Class##PostFix = ::zeno::_defOverloadNodeClassHelper(std::make_unique<Class##PostFix>, #Class, {__VA_ARGS__})

template <class T>
[[deprecated("use ZENO_DEFNODE(T)(...)")]]
inline int defNodeClass(std::string const &id, Descriptor const &desc = {}) {
    getSession().defNodeClass(std::make_unique<T>, id, desc);
    return 1;
}

[[deprecated("use ZENO_DEFNODE(T)(...)")]]
static int _deprecated_ZENDEFNODE_helper() { return 1; }

#define ZENDEFNODE(Class, ...) \
    ZENO_DEFNODE(Class)(__VA_ARGS__), _deprecatedDef##Class = ::zeno::_deprecated_ZENDEFNODE_helper();

}
