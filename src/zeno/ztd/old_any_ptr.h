#pragma once


#include <memory>
#include <cstdint>
#include <variant>
#include <optional>
#include <zeno/ztd/type_traits.h>
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
    std::unique_ptr<T> _M_ptr;

    any_ptr_impl(std::type_info const &type, std::unique_ptr<T> &&ptr)
        : any_ptr_base(type), _M_ptr(std::move(ptr)) {}

    virtual std::shared_ptr<any_ptr_base> clone() const override {
        if constexpr (std::is_copy_constructible_v<T>) {
            return std::make_shared<any_ptr_impl>(
                _M_type, static_cast<T const &>(*_M_ptr));
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

    static T *pointer_cast(type &t) {
        return &t;
    }

    static std::optional<T> value_cast(type const &t) {
        return std::make_optional(t);
    }
};

template <class T>
    requires (variant_contains_v<T, details::numeric_variant<math::vec_dimension_v<T>>>)
struct any_underlying<T> {
    using type = details::numeric_variant<math::vec_dimension_v<T>>;

    static T *pointer_cast(type &t) {
        return std::holds_alternative<T>(t) ? &std::get<T>(t) : nullptr;
    }

    static std::optional<T> value_cast(type const &t) {
        return std::make_optional(std::visit([&] (auto const &v) {
            return (T)v;
        }, t));
    }
};

template <class T>
    requires (true_v<typename T::polymorphic_base_type>)
struct any_underlying<T> {
    using type = typename T::polymorphic_base_type;

    static T *pointer_cast(type &t) {
        return dynamic_cast<T *>(&t);
    }

    static std::optional<T> value_cast(type &t) {
        if (auto p = dynamic_cast<T *>(&t)) {
            return std::make_optional(*p);
        } else {
            return std::nullopt;
        }
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
    explicit any_ptr(std::type_identity<T>, std::unique_ptr<any_underlying_t<T>> &&ptr)
        : _M_base(std::make_shared<details::any_ptr_impl<any_underlying_t<T>>>(
                typeid(T), std::move(ptr)))
    {}

    template <class T>
    any_ptr(std::unique_ptr<T> &&ptr)
        : any_ptr(std::type_identity<T>{}, std::move(ptr))
    {}

    std::type_info const &type() const {
        return _M_base->type();
    }

    any_ptr clone() const {
        return any_ptr(_M_base->clone());
    }

    template <class T>
    T *get() const {
        using U = any_underlying_t<T>;
        auto p = dynamic_cast<details::any_ptr_impl<U> *>(_M_base.get());
        if (!p) return nullptr;
        return any_underlying<T>::pointer_cast(p->_M_val);
    }

    template <class T>
    std::optional<T> cast() const {
        using U = any_underlying_t<T>;
        auto p = dynamic_cast<details::any_ptr_impl<U> *>(_M_base.get());
        if (!p) return std::nullopt;
        return any_underlying<T>::value_cast(p->_M_val);
    }
};

template <class T>
inline any_ptr make_any(auto &&...args) {
    return any_ptr(std::type_identity<T>{},
                   std::make_unique<T>(std::forward<decltype(args)>(args)...));
}

}
}
ZENO_NAMESPACE_END
