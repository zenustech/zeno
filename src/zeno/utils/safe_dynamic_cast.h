#pragma once

#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include <zeno/utils/Error.h>

namespace zeno {

template <class T, class S>
T *safe_dynamic_cast(S *s, std::string const &msg = {}) {
    auto t = dynamic_cast<T *>(s);
    if (!t) {
        throw TypeError(typeid(T), typeid(*s), "safe_dynamic_cast");
    }
    return t;
}

template <class T, class S>
std::shared_ptr<T> safe_dynamic_cast(
        std::shared_ptr<S> s, std::string const &msg = {}) {
    auto t = std::dynamic_pointer_cast<T>(s);
    if (!t) {
        throw TypeError(typeid(T), typeid(*s), "safe_dynamic_cast");
    }
    return t;
}

template <class T>
T safe_any_cast(std::any &&a, std::string const &msg = {}) {
    try {
        return std::any_cast<T>(std::forward<std::any>(a));
    } catch (std::bad_any_cast const &e) {
        throw TypeError(typeid(T), a.type(), "safe_any_cast");
    }
}

}
