#pragma once


#include <variant>
#include <utility>
#include <type_traits>


ZENO_NAMESPACE_BEGIN
namespace ztd {


template <class T>
auto make_monovariant(bool cond, T x) {
    std::variant<std::monostate, T> ret;
    if (cond) ret = std::move(x);
    return ret;
}


template <class T>
concept not_monostate = !std::is_same_v<std::monostate, std::remove_cvref_t<T>>;


}
ZENO_NAMESPACE_END
