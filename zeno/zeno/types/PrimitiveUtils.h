#pragma once

#include <zeno/types/PrimitiveObject.h>

namespace zeno {

static void prim_quads_to_tris(PrimitiveObject *prim) {
    prim->tris.reserve(prim->tris.size() + prim->quads.size() * 2);

    for (auto quad: prim->quads) {
        prim->tris.emplace_back(quad[0], quad[1], quad[2]);
        prim->tris.emplace_back(quad[0], quad[2], quad[3]);
    }
    prim->quads.clear();
}

static void prim_polys_to_tris(PrimitiveObject *prim) {
    prim->tris.reserve(prim->tris.size() + prim->polys.size());

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
    prim->loops.clear();
}

static void prim_polys_to_tris_with_uv(PrimitiveObject *prim) {
    if (!prim->loops.has_attr("uv"))
        return prim_polys_to_tris(prim);

    auto &loop_uv = prim->loops.attr<vec3f>("uv");

    auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
    auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
    auto &uv2 = prim->tris.add_attr<vec3f>("uv2");

    prim->tris.reserve(prim->tris.size() + prim->polys.size());
    uv0.reserve(uv0.size() + prim->polys.size());
    uv1.reserve(uv1.size() + prim->polys.size());
    uv2.reserve(uv2.size() + prim->polys.size());

    for (auto [start, len]: prim->polys) {
        if (len < 3) continue;
        uv0.push_back(loops[start]);
        uv1.push_back(loops[start + 1]);
        uv2.push_back(loops[start + 2]);
        prim->tris.emplace_back(
                prim->loops[start],
                prim->loops[start + 1],
                prim->loops[start + 2]);
        for (int i = 3; i < len; i++) {
            uv0.push_back(loops[start]);
            uv1.push_back(loops[start + i - 1]);
            uv2.push_back(loops[start + i]);
            prim->tris.emplace_back(
                    prim->loops[start],
                    prim->loops[start + i - 1],
                    prim->loops[start + i]);
        }
    }
    prim->loops.clear();
}

}
