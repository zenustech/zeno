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
    std::type_info const &_M_type;

    any_ptr_base(std::type_info const &type) : _M_type(type) {}

    virtual std::shared_ptr<any_ptr_base> clone() const = 0;
    virtual ~any_ptr_base() = default;

    std::type_info const &type() const {
        return _M_type;
    }
};

template <class T>
struct any_ptr_impl : any_ptr_base {
    T _M_val;

    any_ptr_impl(std::type_info const &type, auto &&...args)
        : any_ptr_base(type), _M_val(std::forward<decltype(args)>(args)...) {}

    virtual std::shared_ptr<any_ptr_base> clone() const override {
        if constexpr (std::is_copy_constructible_v<T>) {
            return std::make_shared<any_ptr_impl>(
                _M_type, static_cast<T const &>(_M_val));
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
        return std::visit([&] (auto &v) {
            return (T)v;
        }, t);
    }
};

template <class T>
    requires (true_v<typename T::polymorphic_base_type>)
struct any_underlying<T> {
    using type = typename T::polymorphic_base_type;

    static T *cast(type &t) {
        return dynamic_cast<T *>(&t);
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
                typeid(T), std::forward<decltype(args)>(args)...))
    {
    }

    std::type_info const &type() const {
        return _M_base->type();
    }

    any_ptr clone() const {
        return any_ptr(_M_base->clone());
    }

    template <class T>
    T *cast() const {
        using U = any_underlying_t<T>;
        auto p = dynamic_cast<details::any_ptr_impl<U> *>(_M_base.get());
        if (!p) return nullptr;
        return any_underlying<T>::cast(p->_M_val);
    }
};

template <class T>
inline any_ptr make_any(auto &&...args) {
    return any_ptr(std::type_identity<T>{}, std::forward<decltype(args)>(args)...);
}

}
}
ZENO_NAMESPACE_END
