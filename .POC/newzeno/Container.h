#pragma once

#include "common.h"
#include <cstdint>


namespace zeno::v2::container {

template <class T>
struct is_variant : std::false_type {
};

template <class ...Ts>
struct is_variant<std::variant<Ts...>> : std::true_type {
};

template <class T>
struct is_shared_ptr : std::false_type {
};

template <class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {
    using type = T;
};

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

template <class T, class = void>
struct any_traits {
    using underlying_type = T;
};

template <class T>
struct any_traits<T, std::void_t<decltype(
        std::declval<scalar_type_variant &>() = std::declval<T>()
        )>> {
    using underlying_type = scalar_type_variant;
};

template <class T>
struct any_traits<T *, std::void_t<typename T::polymorphic_base_type>> {
    using underlying_type = typename T::polymorphic_base_type *;
};

template <class T>
struct any_traits<T const *, std::void_t<typename T::polymorphic_base_type>> {
    using underlying_type = typename T::polymorphic_base_type const *;
};

template <class T>
struct any_traits<std::unique_ptr<T>, std::void_t<typename T::polymorphic_base_type>> {
    using underlying_type = std::unique_ptr<typename T::polymorphic_base_type>;
};

template <class T>
struct any_traits<std::shared_ptr<T>, std::void_t<typename T::polymorphic_base_type>> {
    using underlying_type = std::shared_ptr<typename T::polymorphic_base_type>;
};

struct any : std::any {
    any() = default;
    any(any &&a) = default;
    any(any const &a) = default;

    template <class T>
    any(T const &t)
    : std::any(static_cast<typename any_traits<T>::underlying_type>(t))
    {}

    any &operator=(any const &a) = default;

    template <class T>
    any &operator=(T const &t) {
        std::any::operator=(
                static_cast<typename any_traits<T>::underlying_type>(t));
        return *this;
    }
};

struct bad_dynamic_cast : std::bad_cast {
    virtual const char *what() const noexcept { return "bad dynamic_cast"; }
};

template <class T>
T smart_any_cast(any const &a) {
    if constexpr (std::is_same_v<T, any>) {
        return a;
    } else {
        using V = typename any_traits<T>::underlying_type;
        decltype(auto) v = std::any_cast<V const &>(a);
        if constexpr (std::is_pointer_v<T>) {
            auto ret = dynamic_cast<T>(v);
            if (!ret) throw bad_dynamic_cast{};
            return ret;
        } else if constexpr (is_shared_ptr<T>::value) {
            using U = typename is_shared_ptr<T>::type;
            auto ret = std::dynamic_pointer_cast<U>(v);
            if (!ret) throw bad_dynamic_cast{};
            return ret;
        } else if constexpr (is_variant<V>::value && !is_variant<T>::value) {
            return std::visit([] (auto const &x) {
                return (T)x;
            }, v);
        } else {
            return v;
        }
    }
}

template <class T>
T &&exact_any_cast(any &&a) {
    if constexpr (std::is_same_v<T, any>) {
        return a;
    } else {
        using V = typename any_traits<T>::underlying_type;
        decltype(auto) v = std::any_cast<V &&>(a);
        if constexpr (std::is_pointer_v<T>) {
            return static_cast<T>(v);
        } else if constexpr (is_shared_ptr<T>::value) {
            using U = typename is_shared_ptr<T>::type;
            return std::static_pointer_cast<U>(v);
        } else if constexpr (is_variant<V>::value) {
            return std::get<T>(v);
        } else {
            return v;
        }
    }
}

}
