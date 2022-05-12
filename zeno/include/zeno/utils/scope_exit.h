#pragma once

#include <utility>
#include <type_traits>

namespace zeno {

template <class Func>
class scope_exit {
    static_assert(std::is_same_v<std::decay_t<Func>, Func>);

    Func func;
    bool enabled;

public:
    explicit scope_exit(Func &&func) noexcept : func(std::move(func)), enabled(true) {
    }

    bool has_value() const noexcept {
        return enabled;
    }

    void release() noexcept {
        enabled = false;
    }

    void reset() {
        if (enabled) {
            func();
            enabled = false;
        }
    }

    ~scope_exit() noexcept {
        if (enabled)
            func();
    }

    scope_exit(scope_exit const &) = delete;
    scope_exit &operator=(scope_exit const &) = delete;

    scope_exit(scope_exit &&that) noexcept : func(std::move(that.func)), enabled(that.enabled) {
        that.enabled = false;
    }

    scope_exit &operator=(scope_exit &&that) noexcept {
        if (this != &that) {
            enabled = that.enabled;
            that.enabled = false;
            func = std::move(that.func);
        }
    }
};

template <class Func>
scope_exit(Func) -> scope_exit<Func>;


template <class Func>
class scope_enter : public scope_exit<std::invoke_result_t<Func>> {
public:
    explicit scope_enter(Func &&func) noexcept : scope_exit<std::invoke_result_t<Func>>(std::move(func)()) {
    }
};

template <class Func>
scope_enter(Func) -> scope_enter<Func>;


template <class Derived>
class scope_finalizer {
    struct finalize_functor {
        //static_assert(std::is_base_of_v<scope_finalizer, Derived>);

        Derived &that;

        void operator()() const noexcept {
            that._scope_finalize();
        }
    };

    scope_exit<finalize_functor> guard;

public:
    explicit scope_finalizer() noexcept : guard(finalize_functor{static_cast<Derived &>(*this)}) {
    }

    scope_finalizer(scope_finalizer const &) = delete;
    scope_finalizer &operator=(scope_finalizer const &) = delete;

    bool has_value() const noexcept {
        return guard.has_value();
    }

    void release() noexcept {
        return guard.release();
    }

    void reset() noexcept {
        return guard.reset();
    }
};


template <class T>
class scope_restore : public scope_finalizer<scope_restore<T>> {
    T &dst;
    T old;

public:
    scope_restore(T &dst_)
        : dst(dst_), old(std::as_const(dst_)) {
    }

    void _scope_finalize() noexcept {
        dst = std::move(old);
    }
};


template <class T, class U = T, class = std::enable_if_t<!std::is_const_v<T>>>
scope_restore(T &, U &&) -> scope_restore<T>;


template <class T>
class scope_modify : public scope_finalizer<scope_modify<T>> {
    T &dst;
    T old;

public:
    template <class U = T>
    explicit scope_modify(T &dst_, U &&val_) noexcept
        : dst(dst_), old(std::exchange(dst_, std::forward<U>(val_))) {
    }

    void _scope_finalize() noexcept {
        dst = std::move(old);
    }
};


template <class T, class U = T, class = std::enable_if_t<!std::is_const_v<T>>>
scope_modify(T &, U &&) -> scope_modify<T>;

template <class T>
class scope_bind : public scope_finalizer<scope_bind<T>> {
    T dst;

public:
    template <class ...Args>
    explicit scope_bind(T &dst_, Args &&...args) : dst(dst_) {
        dst.bind(std::forward<Args>(args)...);
    }

    void _scope_finalize() {
        dst.unbind();
    }
};

template <class T>
scope_bind(T &) -> scope_bind<T>;

}
