#pragma once

#include <utility>
#include <optional>

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
scope_exit(Func) -> scope_exit<Func>;

template <class Func>
struct optional_scope_exit {
    std::optional<scope_exit<Func>> opt;

    optional_scope_exit(Func &&func) : opt(std::forward<Func>(func)) {
    }

    optional_scope_exit(optional_scope_exit const &) = delete;
    optional_scope_exit &operator=(optional_scope_exit const &) = delete;

    void commit() {
        opt.reset();
    }
};

template <class Func>
optional_scope_exit(Func) -> optional_scope_exit<Func>;

}
