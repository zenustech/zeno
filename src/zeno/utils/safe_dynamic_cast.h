#pragma once

#include <memory>
#include <string>
#include <typeinfo>
#include <zeno/utils/Error.h>

namespace zeno {

template <class T, class S>
T *safe_dynamic_cast(S *s, std::string const &msg = {}) {
    auto t = dynamic_cast<T *>(s);
    if (!t) {
        throw makeError<TypeError>(typeid(T), typeid(*s), "safe_dynamic_cast");
    }
    return t;
}

template <class T, class S>
std::shared_ptr<T> safe_dynamic_cast(
        std::shared_ptr<S> s, std::string const &msg = {}) {
    auto t = std::dynamic_pointer_cast<T>(s);
    if (!t) {
        throw makeError<TypeError>(typeid(T), typeid(*s), "safe_dynamic_cast");
    }
    return t;
}

}
