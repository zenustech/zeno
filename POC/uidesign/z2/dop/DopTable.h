#pragma once


#include <z2/dop/DopFunctor.h>
#include <z2/dop/DopDescriptor.h>


namespace z2::dop {


class DopTable {
    struct Impl {
        ztd::map<std::string, DopFunctor> funcs;
        ztd::map<std::string, DopDescriptor> descs;
    };
    mutable std::unique_ptr<Impl> impl;

    Impl *get_impl() const {
        if (!impl) impl = std::make_unique<Impl>();
        return impl.get();
    }

public:
    std::set<std::string> entry_names() const {
        std::set<std::string> ret;
        for (auto const &[k, v]: get_impl()->funcs) {
            ret.insert(k);
        }
        return ret;
    }

    DopDescriptor const &desc_of(std::string const &name) const {
        return get_impl()->descs.at(name);
    }

    auto const &lookup(std::string const &kind) const {
        return get_impl()->funcs.at(kind);
    }

    int define(std::string const &kind,
               DopDescriptor &&desc, DopFunctor &&func) {
        get_impl()->funcs.emplace(kind, std::move(func));
        return 1;
    }
};


extern DopTable tab;


}  // namespace z2::dop
