#pragma once

#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/utils/type_traits.h>
#include <memory>
#include <optional>

namespace zeno {

template <class T>
T smart_any_cast(std::shared_ptr<IObject> const &p) {
    static_assert(is_shared_ptr<T>::value);
    return safe_dynamic_cast<typename remove_shared_ptr<T>::type>(p);
}

template <class T>
std::optional<T> silent_any_cast() {
    static_assert(is_shared_ptr<T>::value);
    if (auto q = std::dynamic_pointer_cast<typename remove_shared_ptr<T>::type>(p)) {
        return std::make_optional(q);
    } else {
        return std::nullopt;
    }
}

}
