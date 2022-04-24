#pragma once

#include <utility>
#include <type_traits>

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


template <class Derived>
class scope_finalizer {
    struct finalize_functor {
        //static_assert(std::is_base_of_v<scope_finalizer, Derived>);

        Derived &that;

        void operator()() const {
            that._scope_finalize();
        }
    };

    scope_exit<finalize_functor> guard;

public:
    explicit scope_finalizer() : guard(finalize_functor{static_cast<Derived &>(*this)}) {
    }

    scope_finalizer(scope_finalizer const &) = default;
    scope_finalizer &operator=(scope_finalizer const &) = default;

    bool has_value() const {
        return guard.has_value();
    }

    void release() {
        return guard.release();
    }

    void reset() {
        return guard.reset();
    }
};

}
