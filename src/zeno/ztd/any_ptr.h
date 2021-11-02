#pragma once


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
using numeric_auto_vec = std::conditional_t<N == 0,
      T, math::vec<std::max(N, (size_t)1), T>>;

template <size_t N = 0>
using numeric_variant = std::variant
    < numeric_auto_vec<N, bool>
    , numeric_auto_vec<N, int32_t>
    , numeric_auto_vec<N, float>
    >;

}

template <class T>
struct any_underlying {
    using type = T;

    static std::shared_ptr<T> pointer_cast(std::shared_ptr<void> &&t) {
        return std::static_pointer_cast<T>(std::move(t));
    }

    static std::optional<T> value_cast(type const &t) {
        return std::make_optional(t);
    }
};

template <class T>
    requires (variant_contains_v<T, details::numeric_variant<math::vec_dimension_v<T>>>)
struct any_underlying<T> {
    using type = details::numeric_variant<math::vec_dimension_v<T>>;

    static std::shared_ptr<T> pointer_cast(std::shared_ptr<void> &&t) {
        auto &v = *std::static_pointer_cast<type>(std::move(t));
        return std::holds_alternative<T>(v) ? stale_shared(&std::get<T>(v)) : nullptr;
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

    static std::shared_ptr<T> pointer_cast(std::shared_ptr<void> &&t) {
        return std::dynamic_pointer_cast<T>(std::static_pointer_cast<type>(std::move(t)));
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

class any_ptr : public std::shared_ptr<void> {
private:
    std::type_info const *_M_type{&typeid(void)};
    std::type_info const *_M_utype{&typeid(void)};

public:
    any_ptr() = default;
    any_ptr(any_ptr const &) = default;
    any_ptr &operator=(any_ptr const &) = default;
    any_ptr(any_ptr &&) = default;
    any_ptr &operator=(any_ptr &&) = default;

    template <class T>
    any_ptr(std::in_place_t, T const &val)
        : std::shared_ptr<void>(std::make_shared<any_underlying_t<T>>(val))
        , _M_type(&typeid(T))
        , _M_utype(&typeid(any_underlying_t<T>))
    {}

    template <class T>
    any_ptr(std::shared_ptr<T> &&ptr)
        : std::shared_ptr<void>(std::move(ptr))
        , _M_type(&typeid(*ptr))
        , _M_utype(&typeid(any_underlying_t<T>))
    {}

    std::type_info const &type() const {
        return *_M_type;
    }

    std::type_info const &utype() const {
        return *_M_utype;
    }

    template <class T>
    std::shared_ptr<T> pointer_cast() const {
        using U = any_underlying_t<T>;
        if (typeid(U) != *_M_utype) return nullptr;
        auto p = std::static_pointer_cast<U>(*this);
        return any_underlying<T>::pointer_cast(std::move(p));
    }

    template <class T>
    std::optional<T> value_cast() const {
        using U = any_underlying_t<T>;
        if (typeid(U) != *_M_utype) return std::nullopt;
        auto p = static_cast<U *>(this->get());
        return any_underlying<T>::value_cast(*p);
    }
};

inline any_ptr make_any(auto const &t) {
    return any_ptr(std::in_place, t);
}

template <class T>
inline std::shared_ptr<T> pointer_cast(any_ptr p) {
    [[likely]] if (auto q = p.pointer_cast<T>()) {
        return q;
    } else {
        throw ztd::format_error("TypeError: pointer_cast failed from {} to {}",
                    cpp_type_name(p.type()), cpp_type_name(typeid(T)));
    }
}

template <class T>
inline T value_cast(any_ptr p) {
    [[likely]] if (auto q = p.value_cast<T>()) {
        return *q;
    } else {
        throw ztd::format_error("TypeError: value_cast failed from {} to {}",
                    cpp_type_name(p.type()), cpp_type_name(typeid(T)));
    }
}

}
}
ZENO_NAMESPACE_END
