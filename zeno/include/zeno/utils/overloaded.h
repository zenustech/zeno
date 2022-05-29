#pragma once

#include <variant>
#include <utility>

template <class ...Fs>
struct overloaded : Fs... {
    using Fs::operator()...;

    struct __auto_guess {};

    template <class Ret = __auto_guess, class ...Ts>
    decltype(auto) match(Ts &&...ts) const {
        if constexpr (std::is_same_v<Ret, __auto_guess>) {
            return std::visit(*this, std::forward<Ts>(ts)...);
        } else {
            return std::visit([this] (auto &&...ts) -> Ret {
                return (*this)(std::forward<decltype(ts)>(ts)...);
            }, std::forward<Ts>(ts)...);
        }
    }
};

template <class ...Fs>
overloaded(Fs...) -> overloaded<Fs...>;

