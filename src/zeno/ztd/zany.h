#pragma once


#include <any>
#include <memory>
#include <cstdint>
#include <variant>
#include <optional>
#include <zeno/ztd/type_traits.h>
#include <zeno/ztd/type_info.h>
#include <zeno/ztd/error.h>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace ztd {

namespace zany_details {

template <size_t N, class T>
using auto_vec = std::conditional_t<N == 0,
      T, math::vec<std::max(N, (size_t)1), T>>;

template <size_t N>
using scalar_variant = std::variant
    < auto_vec<N, bool>
    , auto_vec<N, int8_t>
    , auto_vec<N, int16_t>
    , auto_vec<N, int32_t>
    , auto_vec<N, int64_t>
    , auto_vec<N, uint8_t>
    , auto_vec<N, uint16_t>
    , auto_vec<N, uint32_t>
    , auto_vec<N, uint64_t>
    , auto_vec<N, float>
    , auto_vec<N, double>
    >;

template <class T>
struct vector_dimension {
};

template <class T>
    requires (variant_contains_v<T, scalar_variant<0>>)
struct vector_dimension<T> : std::integral_constant<size_t, 0> {
};

template <size_t N, class T>
    requires (variant_contains_v<T, scalar_variant<0>>)
struct vector_dimension<math::vec<N, T>> : std::integral_constant<size_t, N> {
};

}


template <class T>
struct zany_underlying {
    using type = T;

    static std::optional<T> convert(type const &t) {
        return std::make_optional(t);
    }
};

template <class T>
    requires (variant_contains_v<T, zany_details::scalar_variant<zany_details::vector_dimension<T>::value>>)
struct zany_underlying<T> {
    using type = zany_details::scalar_variant<zany_details::vector_dimension<T>::value>;

    static std::optional<T> convert(type const &t) {
        if (std::holds_alternative<T>(t)) {
            return std::make_optional(std::get<T>(t));
        } else {
            return std::nullopt;
        }
    }
};

template <class T>
    requires (true_v<typename T::polymorphic_base_type>)
struct zany_underlying<std::shared_ptr<T>> {
    using type = std::shared_ptr<typename T::polymorphic_base_type>;

    static std::optional<T> convert(type const &t) {
        if (auto p = std::dynamic_pointer_cast<T>(t)) {
            return std::make_optional(p);
        } else {
            return std::nullopt;
        }
    }
};


template <class T>
using zany_underlying_t = typename zany_underlying<std::decay_t<T>>::type;

template <class T>
static constexpr auto zany_underlying_f = zany_underlying<std::decay_t<T>>::convert;


struct zany {
    std::any _M_any;

    zany() = default;
    zany(zany const &) = default;
    zany &operator=(zany const &) = default;
    zany(zany &&) = default;
    zany &operator=(zany &&) = default;

    template <class T>
        requires (!std::is_same_v<T, std::any> && !std::is_same_v<T, zany>)
    std::optional<T> cast() const {
        if (_M_any.type() != typeid(zany_underlying_t<T>))
            return std::nullopt;
        return zany_underlying_f<T>(std::any_cast<zany_underlying_t<T>>(_M_any));
    }

    template <class T>
        requires (!std::is_same_v<T, std::any> && !std::is_same_v<T, zany>)
    explicit operator T() const {
        auto o = cast<T>();
        [[unlikely]] if (!o.has_value())
            throw format_error("TypeError: cannot cast {} -> {}",
                               cpp_type_name(_M_any.type()), cpp_type_name(typeid(T)));
        return o.value();
    }

    template <class T>
        requires (!std::is_same_v<T, std::any> && !std::is_same_v<T, zany>)
    zany(T const &t)
        : _M_any(zany_underlying_t<T>(t))
    {}

    template <class T>
        requires (!std::is_same_v<T, std::any> && !std::is_same_v<T, zany>)
    zany &operator=(T const &t) {
        _M_any = zany_underlying_t<T>(t);
        return *this;
    }

    bool has_value() const {
        return _M_any.has_value();
    }

    decltype(auto) type() const {
        return _M_any.type();
    }
};

template <class T>
T zany_cast(zany const &a) {
    return a.operator T();
}

}
ZENO_NAMESPACE_END
