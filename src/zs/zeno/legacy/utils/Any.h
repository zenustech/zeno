#pragma once

#include "vec.h"
#include <zeno2/ztd/zany.h>
#include "type_traits.h"


namespace zeno {

using scalar_type_variant = std::variant
        < bool
        , uint8_t
        , uint16_t
        , uint32_t
        , uint64_t
        , int8_t
        , int16_t
        , int32_t
        , int64_t
        , float
        , double
        >;

template <size_t N>
using vector_type_variant = std::variant
        < vec<N, bool>
        , vec<N, uint8_t>
        , vec<N, uint16_t>
        , vec<N, uint32_t>
        , vec<N, uint64_t>
        , vec<N, int8_t>
        , vec<N, int16_t>
        , vec<N, int32_t>
        , vec<N, int64_t>
        , vec<N, float>
        , vec<N, double>
        >;

template <class T>
using any_underlying_type_t = zeno2::ztd::zany_underlying_t<T>;

using Any = zeno2::ztd::zany;

template <class T>
std::optional<T> exact_any_cast(Any a) {
    if constexpr (std::is_same_v<std::decay_t<T>, Any>) {
        return std::make_optional(a);
    } else {
        using V = any_underlying_type_t<T>;
        if (typeid(V) != a.type()) return std::nullopt;
        decltype(auto) v = std::any_cast<V const &>(a);
        if constexpr (is_shared_ptr<T>::value) {
            using U = typename remove_shared_ptr<T>::type;
            decltype(auto) ptr = std::static_pointer_cast<U>(std::move(v));
            if (!ptr) return std::nullopt;
            return std::make_optional(std::move(ptr));
        } else if constexpr (is_variant<V>::value && !is_variant<T>::value) {
            if (!std::holds_alternative<T>(v)) return std::nullopt;
            return std::make_optional(std::get<T>(v));
        } else {
            return v;
        }
    }
}

template <class T>
std::optional<T> silent_any_cast(Any const &a) {
    if constexpr (std::is_same_v<T, Any>) {
        return std::make_optional(a);

    } else if constexpr (is_variant<T>::value) {
    using TupleT = typename variant_to_tuple<T>::type;
    T v;
    if (static_for<0, std::tuple_size_v<TupleT>>([&] (auto i) {
        using Ti = std::tuple_element_t<i.value, TupleT>;
        auto o = exact_any_cast<Ti>(std::move(a));
        if (o.has_value()) {
            v = o.value();
            return true;
        }
        return false;
    })) return std::make_optional(v);
    else return std::nullopt;

    } else {
        using V = any_underlying_type_t<T>;
        if (typeid(V) != a.type()) return std::nullopt;
        decltype(auto) v = std::any_cast<V const &>(a);
        if constexpr (is_shared_ptr<T>::value) {
            using U = typename remove_shared_ptr<T>::type;
            auto ptr = std::dynamic_pointer_cast<U>(v);
            if (!ptr) return std::nullopt;
            return std::make_optional(ptr);
        } else if constexpr (is_variant<V>::value && !is_variant<T>::value) {
            return std::make_optional(std::visit([] (auto const &x) {
                return (T)x;
            }, v));
        } else {
            return std::make_optional(v);
        }
    }
}

using zany = Any;

}
