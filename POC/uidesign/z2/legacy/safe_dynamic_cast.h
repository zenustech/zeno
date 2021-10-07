#pragma once

#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include "Any.h"
#include "Exception.h"

namespace z2::legacy {

template <class T, class S>
T *safe_dynamic_cast(S *s, std::string const &msg = {}) {
    auto t = dynamic_cast<T *>(s);
    if (!t) {
        throw Exception(msg + "expect `"
                + typeid(T).name() + "`, got `"
                + typeid(*s).name() + "` (safe_dynamic_cast)");
    }
    return t;
}

template <class T, class S>
std::shared_ptr<T> safe_dynamic_cast(
        std::shared_ptr<S> s, std::string const &msg = {}) {
    auto t = std::dynamic_pointer_cast<T>(s);
    if (!t) {
        throw Exception(msg + "expect `"
                + typeid(T).name() + "`, got `"
                + typeid(*s).name() + "` (safe_dynamic_cast)");
    }
    return t;
}

template <class T>
T safe_any_cast(std::any &&a, std::string const &msg = {}) {
    try {
        return std::any_cast<T>(std::forward<std::any>(a));
    } catch (std::bad_any_cast const &e) {
        throw Exception(msg + "expect `"
                + typeid(T).name() + "`, got `"
                + a.type().name() + "` (safe_any_cast for std::any)");
    }
}

template <class T>
T safe_any_cast(Any const &a, std::string const &msg = {}) {
    if (auto o = silent_any_cast<T>(a); o.has_value()) {
        return o.value();
    } else {
        throw Exception(msg + "expect `"
                + typeid(T).name() + "`, got `"
                + a.type().name() + "` (safe_any_cast)");
    }
}

}
