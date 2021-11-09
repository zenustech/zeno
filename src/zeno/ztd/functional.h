#pragma once


#include <zeno/common.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _H_functional {


/*template <class T = void, class ...Ts>
using promise = std::function<T(Ts...)>;

template <class ...Ts, class T>
promise<T, Ts...> make_promise(T val) {
    return [val = std::move(val)] (Ts...) -> T { return val; };
}*/


struct dtor_function : std::function<void()> {
    using std::function<void()>::function;

    dtor_function(dtor_function const &) = delete;
    dtor_function &operator=(dtor_function const &) = delete;
    dtor_function(dtor_function &&) = default;
    dtor_function &operator=(dtor_function &&) = default;

    ~dtor_function() {
        if (std::function<void()>::operator bool())
            std::function<void()>::operator()();
    }
};


template <class ...Fs>
struct overloaded : private Fs... {
    overloaded(Fs &&...fs)
        : Fs(std::forward<Fs>(fs))... {}

    using Fs::operator()...;
};


}
}
ZENO_NAMESPACE_END
