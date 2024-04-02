#pragma once

#include <zeno/core/Session.h>

namespace zeno {

// deprecated
//template <class F>
//auto _defOverloadNodeClassHelper(F const &func, std::string const &name, std::vector<std::string> const &types) {
    //return [=] (zeno::Descriptor const &desc) -> int {
        //getSession().defOverloadNodeClass(func, name, types, desc);
        //return 1;
    //};
//}

#define ZENO_DEFNODE2(Class) \
    static struct _Def##Class { \
        _Def##Class(::zeno::CustomUI const &desc) {\
            ::zeno::getSession().defNodeClass2([] () -> std::shared_ptr<::zeno::INode> { \
                return std::make_shared<Class>(); }, #Class, desc); \
        } \
    } _def##Class


#define ZENO_DEFNODE(Class) \
    static struct _Def##Class { \
        _Def##Class(::zeno::Descriptor const &desc) { \
            ::zeno::getSession().defNodeClass([] () -> std::shared_ptr<::zeno::INode> { \
                return std::make_shared<Class>(); }, #Class, desc); \
        } \
    } _def##Class

// deprecated:
template <class T>
[[deprecated("use ZENO_DEFNODE(T)(...)")]]
inline int defNodeClass(std::string const &id, Descriptor const &desc = {}) {
    getSession().defNodeClass([] () -> std::shared_ptr<INode> { return std::make_shared<T>(); }, id, desc);
    return 1;
}

// deprecated:
#define ZENDEFNODE(Class, ...) \
    ZENO_DEFNODE(Class)(__VA_ARGS__);

#define ZENDEFINE(Class, ...) \
    ZENO_DEFNODE2(Class)(__VA_ARGS__);

// deprecated:
#define ZENO_DEFOVERLOADNODE(Class, PostFix, ...) \
    static int _deprecatedDefOverload##Class##And##PostFix = [] (::zeno::Descriptor const &) { return 1; }
//#define ZENO_DEFOVERLOADNODE(Class, PostFix, ...) \
    //static int _def##Class##PostFix = ::zeno::_defOverloadNodeClassHelper(std::make_unique<Class##PostFix>, #Class, {__VA_ARGS__})

}
