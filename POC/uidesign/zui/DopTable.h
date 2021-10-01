#pragma once


#include "DopContext.h"


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
} tab;


static int def_readvdb = tab.define("readvdb", [] (DopContext *ctx) {
    //auto dx = std::any_cast<float>(ctx.in[0]);
    //printf("readvdb %f\n", dx);
    printf("readvdb\n");
});

static int def_vdbsmooth = tab.define("vdbsmooth", [] (DopContext *ctx) {
    printf("vdbsmooth\n");
});

static int def_vdberode = tab.define("vdberode", [] (DopContext *ctx) {
    printf("vdberode\n");
});
