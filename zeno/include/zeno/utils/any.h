#pragma once

#include <zeno/utils/Exception.h>
#include <zinc/any.h>


namespace zeno {

using namespace zinc;

template <class T>
T smart_any_cast(any const &a) {
    auto o = implicit_any_cast<T>(a);
    if (!o.has_value()) {
        throw Exception(msg + "expect `"
                + typeid(T).name() + "`, got `"
                + a.type().name() + "` (smart_any_cast)");
    }
    return o.value();
}

}
