#pragma once

#include <variant>
#include <type_traits>

namespace zeno {

static std::variant<std::false_type, std::true_type> boolean_variant(bool val) {
    if (val) return std::true_type{};
    else return std::false_type{};
}

template <class Func>
auto boolean_switch(bool val, Func &&func) {
    return std::visit(std::forward<Func>(func), boolean_variant(val));
}

}
