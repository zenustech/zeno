#pragma once


#include "DopFunctor.h"


struct DopTable {
    struct Impl {
        ztd::Map<std::string, DopFunctor> funcs;
    };
    mutable std::unique_ptr<Impl> impl;

    Impl *get_impl() const {
        if (!impl) impl = std::make_unique<Impl>();
        return impl.get();
    }

    auto const &lookup(std::string const &kind) const {
        return get_impl()->funcs.at(kind);
    }

    int define(std::string const &kind, DopFunctor &&func) {
        get_impl()->funcs.emplace(kind, std::move(func));
        return 1;
    }
};


extern DopTable tab;
