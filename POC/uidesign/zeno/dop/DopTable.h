#pragma once


#include <zeno/dop/DopFunctor.h>


namespace zeno::dop {


struct DopTable {
    struct Impl {
        ztd::map<std::string, DopFunctor> funcs;
    };
    mutable std::unique_ptr<Impl> impl;

    std::set<std::string> entry_names() const {
        std::set<std::string> ret;
        for (auto const &[k, v]: get_impl()->funcs) {
            ret.insert(k);
        }
        return ret;
    }

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


}  // namespace zeno::dop
