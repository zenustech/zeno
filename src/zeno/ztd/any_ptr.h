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
inline namespace _any_ptr_h {

namespace details {

template <size_t N, class T>
using auto_vec = std::conditional_t<N == 0,
      T, math::vec<std::max(N, (size_t)1), T>>;

template <size_t N = 0>
using numeric_variant = std::variant
    < auto_vec<N, bool>
    , auto_vec<N, int32_t>
    , auto_vec<N, float>
    >;

struct any_ptr_base {
    virtual std::shared_ptr<any_ptr_base> clone() const = 0;
    virtual ~any_ptr_base() = default;
};

template <class T>
struct any_ptr_impl : any_ptr_base {
    T _M_t;

    any_ptr_impl(std::in_place_t, auto &&...args)
        : _M_t(std::forward<decltype(args)>(args)...) {}

    virtual std::shared_ptr<any_ptr_base> clone() const override {
        if constexpr (std::is_copy_constructible_v<T>) {
            return std::make_shared<any_ptr_impl>(
                std::in_place, static_cast<T const &>(_M_t));
        } else {
            return nullptr;
        }
    }

    virtual ~any_ptr_impl() = default;
};
}

template <class T>
struct any_underlying {
    using type = T;

    static T *cast(type &t) {
        return &t;
    }
};

template <class T>
    requires (variant_contains_v<details::numeric_variant<math::vec_dimension_v<T>>, T>)
struct any_underlying<T> {
    using type = details::numeric_variant<math::vec_dimension_v<T>>;

    static T *cast(type &t) {
        return std::holds_alternative<T>(t) ? &std::get<T>(t) : nullptr;
    }
};

template <class T>
using any_underlying_t = typename any_underlying<T>::type;

struct any_ptr {
private:
    std::shared_ptr<details::any_ptr_base> _M_base;

    explicit any_ptr(std::shared_ptr<details::any_ptr_base> &&base)
        : _M_base(std::move(base)) {}

public:
    any_ptr() = default;
    any_ptr(any_ptr const &) = default;
    any_ptr &operator=(any_ptr const &) = default;
    any_ptr(any_ptr &&) = default;
    any_ptr &operator=(any_ptr &&) = default;

    template <class T>
    explicit any_ptr(std::type_identity<T>, auto &&...args)
        : _M_base(std::make_shared<details::any_ptr_impl<any_underlying_t<T>>>(
                std::in_place, std::forward<decltype(args)>(args)...))
    {
    }

    any_ptr clone() const {
        return any_ptr(_M_base->clone());
    }

    template <class T>
    T *cast() const {
        using U = any_underlying_t<T>;
        auto p = dynamic_cast<details::any_ptr_impl<U> *>(_M_base.get());
        if (!p) return nullptr;
        return any_underlying<T>::cast(p->_M_t);
    }

    template <class T>
    std::optional<T> get() const {
        auto p = cast<T>();
        return p ? std::make_optional(p) : std::nullopt;
    }
};

template <class T>
inline any_ptr make_any(auto &&...args) {
    return any_ptr(std::type_identity<T>{}, std::forward<decltype(args)>(args)...);
}

}
}
ZENO_NAMESPACE_END
