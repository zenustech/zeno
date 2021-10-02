#pragma once


#include "stdafx.h"


class DopLazy {
    struct Impl {
        std::function<std::any()> fun;
        std::any val{};

        template <class F>
        Impl(F fun) : fun(fun) {}
    };

    std::shared_ptr<Impl> impl;

public:
    DopLazy() = default;
    DopLazy(DopLazy const &) = default;
    DopLazy &operator=(DopLazy const &) = default;
    DopLazy(DopLazy &&) = default;
    DopLazy &operator=(DopLazy &&) = default;

    template <class F>
    DopLazy(F fun) : impl(std::make_shared<Impl>(std::move(fun))) {}

    template <class F>
    DopLazy &operator=(F fun) {
        impl = std::make_shared<Impl>(std::move(fun));
        return *this;
    }

    std::any operator()() const {
        if (!impl) {
            throw ztd::makeException("null dop lazy called");
        }
        if (!impl->val.has_value()) {
            printf("reval\n");
            printf("going %p\n", impl.get());
            impl->val = impl->fun();
            printf("okay\n");
        }
        return impl->val;
    }

    DopLazy const &reset() const {
        impl->val = {};
        return *this;
    }
};
