#pragma once


#include "stdafx.h"


class DopLazy {
    struct Impl {
        std::function<std::any()> fun;
        std::any val;

        template <class F>
        Impl(F &&fun) : fun(fun) {}
    };

    std::shared_ptr<Impl> impl;

public:
    template <class F>
    DopLazy(F &&fun) : impl(std::make_shared<Impl>(std::move(fun))) {}

    DopLazy() = default;
    DopLazy(DopLazy const &) = default;
    DopLazy &operator=(DopLazy const &) = default;
    DopLazy(DopLazy &&) = default;
    DopLazy &operator=(DopLazy &&) = default;

    std::any operator()() const {
        if (!impl->val.has_value()) {
            impl->val = impl->fun();
        }
        return impl->val;
    }

    DopLazy const &reset() const {
        impl->val = {};
        return *this;
    }
};
