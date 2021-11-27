#pragma once

#include <zeno/types/PrimitiveObject.h>

namespace zeno {

static void prim_triangulate(PrimitiveObject *prim) {
    prim->tris.clear();

    for (auto [start, len]: prim->polys) {
        if (len < 3) continue;
        prim->tris.emplace_back(
                prim->loops[start],
                prim->loops[start + 1],
                prim->loops[start + 2]);
        for (int i = 3; i < len; i++) {
            prim->tris.emplace_back(
                    prim->loops[start],
                    prim->loops[start + i - 1],
                    prim->loops[start + i]);
        }
    }
}

// makeXinxinVeryHappy
static auto primGetVal(PrimitiveObject *prim, size_t i) {
    std::map<std::string, std::variant<vec3f, float>> ret;
    prim->foreach_attr([&] (auto const &name, auto const &arr) {
        ret.emplace(name, arr[i]);
    });
    return ret;
}

static void primAppendVal(PrimitiveObject *prim, PrimitiveObject *primB, size_t i) {
    primB->foreach_attr([&] (auto const &name, auto const &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        prim->attr<T>(name).push_back(arr[i]);
    });
}

}
