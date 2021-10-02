#pragma once


#include "stdafx.h"


class DopLazy {
    std::function<std::any()> fun;
    std::any val;

public:
    template <class F>
    DopLazy(F &&fun) : fun(std::move(fun)) {}

    DopLazy() = default;
    DopLazy(DopLazy const &) = default;
    DopLazy &operator=(DopLazy const &) = default;
    DopLazy(DopLazy &&) = default;
    DopLazy &operator=(DopLazy &&) = default;

    std::any operator()() {
        if (!val.has_value()) {
            val = fun();
        }
        return val;
    }

    DopLazy &reset() {
        val = {};
        return *this;
    }
};
