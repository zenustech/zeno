#pragma once


#include <z2/dop/DopDescriptor.h>


namespace z2::dop {


class DopTable {
    struct Impl {
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
        for (auto const &[k, v]: get_impl()->descs) {
            ret.insert(k);
        }
        return ret;
    }

    DopDescriptor const &desc_of(std::string const &kind) const {
        return get_impl()->descs.at(kind);
    }

    DopFunctor const &lookup(std::string const &kind) const {
        return desc_of(kind).func;
    }

    int define(std::string const &kind, DopDescriptor &&desc) {
        get_impl()->descs.emplace(kind, std::move(desc));
        return 1;
    }
};


extern DopTable tab;


}  // namespace z2::dop
