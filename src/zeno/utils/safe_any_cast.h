#pragma once

#include <memory>
#include <string>
#include <typeinfo>
#include <zeno/utils/Error.h>

namespace zeno {

template <class T>
T safe_any_cast(std::any &&a, std::string const &msg = {}) {
    try {
        return std::any_cast<T>(std::forward<std::any>(a));
    } catch (std::bad_any_cast const &e) {
        throw TypeError(typeid(T), a.type(), "safe_any_cast");
    }
}

}
