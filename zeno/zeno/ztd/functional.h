#pragma once


#include <zeno/common.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _H_functional {


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
