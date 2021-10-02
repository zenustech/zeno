#pragma once


#include "stdafx.h"


#if 0
class DopLazy {
    struct Impl {
        std::function<std::any()> fun;
        std::any val;

        template <class F>
        Impl(F fun)
            : fun(std::move(fun))
        {}
    };

    std::shared_ptr<Impl> impl;

public:
    DopLazy() = default;
    DopLazy(DopLazy const &) = default;
    DopLazy &operator=(DopLazy const &) = default;
    DopLazy(DopLazy &&) = default;
    DopLazy &operator=(DopLazy &&) = default;

    template <class F>
    DopLazy(F fun)
        : impl(std::make_shared<Impl>(std::move(fun)))
    {
    }

    std::any operator()() const {
        if (!impl) {
            throw ztd::makeException("null dop lazy called");
        }
        if (!impl->val.has_value()) {
            impl->val = impl->fun();
        }
        return impl->val;
    }

    DopLazy const &reset() const {
        impl->val = {};
        return *this;
    }

    bool has_value() const {
        return impl != nullptr;
    }
};
#else
using DopLazy = std::any;
#endif
