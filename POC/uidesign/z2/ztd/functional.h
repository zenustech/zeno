#pragma once


#include <functional>


namespace z2::ztd {


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
struct overloaded : Fs... {
    overloaded(Fs &&...fs)
        : Fs(std::forward<Fs>(fs))... {}

    using Fs::operator()...;
};

template <class ...Fs>
overloaded(Fs &&...) -> overloaded<Fs...>;


};
