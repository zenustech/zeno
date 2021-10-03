#pragma once


#include <functional>


namespace z2::ztd {


template <class Ret = void, class ...Args>
using function_r = std::function<Ret(Args...)>;


struct dtor_function : std::function<void()> {
    using std::function<void()>::function;

    dtor_function(dtor_function const &) = delete;
    dtor_function &operator=(dtor_function const &) = delete;
    dtor_function(dtor_function &&) = default;
    dtor_function &operator=(dtor_function &&) = default;

    ~dtor_function() {
        std::function<void()>::operator()();
    }
};


template <class ...Fs>
struct overloaded : private Fs... {
    overloaded(Fs &&...fs)
        : Fs(std::forward<Fs>(fs))... {}

    using Fs::operator()...;
};

template <class ...Fs>
overloaded(Fs &&...) -> overloaded<Fs...>;


};
