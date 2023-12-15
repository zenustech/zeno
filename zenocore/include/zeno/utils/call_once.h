#pragma once

#include <utility>
#include <optional>
#include <zeno/utils/type_traits.h>

namespace zeno {

template <class T>
class once_toggle {
    bool armed;
    T t1;
    T t2;

public:
    once_toggle(T &&t1, T &&t2)
        : armed(false)
        , t1(std::move(t1))
        , t2(std::move(t2))
    {
    }

    constexpr operator bool() const {
        return !armed;
    }

    constexpr T &operator()() {
        if (armed) {
            t1 = std::move(t2);
            armed = false;
        }
        return t1;
    }
};

template <class T>
once_toggle(T, T) -> once_toggle<T>;


template <class F>
class once_cached {
    F f;
    using Ret = std::decay_t<std::invoke_result_t<F>>;
    std::optional<avoid_void_t<Ret>> ret;

public:
    once_cached(F &&f) : f(std::move(f))
    {
    }

    constexpr operator bool() const {
        return ret.has_value();
    }

    constexpr decltype(auto) operator()() {
        if (!ret.has_value()) {
            ret.emplace(avoid_void_call(std::move(f)));
        }
        if constexpr (std::is_void_v<Ret>)
            return;
        else
            return *ret;
    }
};

template <class F>
once_cached(F) -> once_cached<F>;

}
