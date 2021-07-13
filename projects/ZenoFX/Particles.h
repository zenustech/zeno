#pragma once

#include <zeno/zeno.h>
#include "Array.h"
#include <string>

struct Particles : zeno::IObject {
    std::map<std::string, VariantArray> attrs;

    template <class F>
    void foreach_attr(F const &f) {
        for (auto &[key, arr]: attrs) {
            std::visit([&key, &f](auto &arr) {
                f(key, arr);
            }, arr);
        }
    }
};
