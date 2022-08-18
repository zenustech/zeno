#pragma once

#include <variant>
#include <utility>

namespace zeno {

template <class ...Fs>
struct overloaded : protected Fs... {
    overloaded(Fs ...fs) : Fs(std::move(fs))... {
    }

    using Fs::operator()...;

    template <class ...Ts>
    decltype(auto) match(Ts &&...ts) const {
        return std::visit(*this, std::forward<Ts>(ts)...);
    }

    template <class Ret, class ...Ts>
    decltype(auto) match(Ts &&...ts) const {
        return std::visit([this] (auto &&...ts) -> Ret {
            return (*this)(std::forward<decltype(ts)>(ts)...);
        }, std::forward<Ts>(ts)...);
    }
};

template <class ...Fs>
overloaded(Fs...) -> overloaded<Fs...>;

}
