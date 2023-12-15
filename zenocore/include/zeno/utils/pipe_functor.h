#pragma once

#include <zeno/utils/api.h>

namespace zeno {

template <class Func>
class pipe_functor {
    Func func;

public:
    explicit pipe_functor() = default;

    template <class FuncU = Func>
    constexpr explicit pipe_functor(FuncU &&funcU) noexcept : func(std::forward<FuncU>(funcU)) {}

    template <class Arg>
    constexpr decltype(auto) operator|(Arg &&arg) const noexcept(noexcept(func(std::forward<Arg>(arg)))) {
        return func(std::forward<Arg>(arg));
    }
};

template <class T>
class pipe_constructor {
    explicit pipe_constructor() = default;

    template <class Arg>
    constexpr T operator|(Arg &&arg) const noexcept(noexcept(T(std::forward<Arg>(arg)))) {
        return T(std::forward<Arg>(arg));
    }
};

template <class T>
inline constexpr pipe_constructor<T> pipe_construct{};

}
