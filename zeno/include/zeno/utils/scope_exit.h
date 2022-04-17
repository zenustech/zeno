#pragma once

#include <utility>

namespace zeno {

template <class Func>
class scope_exit {
    Func func;
    bool enabled;

public:
    scope_exit(Func &&func) : func(std::forward<Func>(func)), enabled(true) {
    }

    bool has_value() const {
        return enabled;
    }

    void release() {
        enabled = false;
    }

    void reset() {
        if (enabled) {
            func();
            enabled = false;
        }
    }

    ~scope_exit() {
        if (enabled)
            func();
    }

    scope_exit(scope_exit const &) = delete;
    scope_exit &operator=(scope_exit const &) = delete;

    scope_exit(scope_exit &&that) : func(std::move(func)), enabled(that.enabled) {
        that.enabled = false;
    }

    scope_exit &operator=(scope_exit &&that) {
        if (this != &that) {
            enabled = that.enabled;
            that.enabled = false;
            func = std::move(that.func);
        }
    }
};

template <class Func>
scope_exit(Func) -> scope_exit<Func>;

}
