#pragma once

#include <utility>

namespace zeno {

template <class Func>
struct scope_exit {
    Func func;

    scope_exit(Func &&func) : func(std::forward<Func>(func)) {
    }

    ~scope_exit() {
        func();
    }

    scope_exit(scope_exit const &) = delete;
    scope_exit &operator=(scope_exit const &) = delete;
    scope_exit(scope_exit &&) = delete;
    scope_exit &operator=(scope_exit &&) = delete;
};

template <class Func>
scope_exit(Func &&) -> scope_exit<Func>;

}
