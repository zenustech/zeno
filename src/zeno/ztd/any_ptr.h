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

struct any_ptr_base {
    virtual any_ptr_base clone() const = 0;
    virtual ~any_ptr_base() = default;
};

template <class T>
struct any_ptr_impl : any_ptr_base {
    std::shared_ptr<T> _M_t;

    any_ptr_impl(std::shared_ptr<T> &&t)
        : _M_t(std::move(t)) {}

    virtual any_ptr_base clone() const override {
        if constexpr (std::is_copy_constructible_v<T>) {
            return std::make_shared<T>(static_cast<T const &>(*_M_t));
        } else {
            return nullptr;
        }
    }

    virtual ~any_ptr_impl() = default;
};

}

template <class T>
struct any_ptr {
    std::shared_ptr<details::any_ptr_base> _M_base;

    explicit any_ptr(std::shared_ptr<T> &&t)
        : _M_base(std::make_shared<details::any_ptr_impl<T>>(std::move(t)))
    {
    }
};

template <class T, class U = any_underlying_t<T>>
inline auto make_any(auto &&args) {
    return any_ptr<U>(std::make_shared<U>(std::forward<decltype(args)>(args)...));
}

}
}
}
ZENO_NAMESPACE_END
