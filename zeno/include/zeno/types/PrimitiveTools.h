#pragma once

#include <zeno/types/PrimitiveObject.h>

namespace zeno {

static void prim_triangulate(PrimitiveObject *prim) {
    prim->resize(prim->verts().size());
    prim->points.clear();
    prim->lines.clear();
    prim->tris.clear();

    for (auto [start, len]: prim->polys()) {
        if (len == 1) {
            prim->points.emplace_back(
                prim->loops()[start]);
        } else if (len == 2) {
            prim->lines.emplace_back(
                prim->loops()[start],
                prim->loops()[start + 1]);
        } else if (len > 3) {
            prim->tris.emplace_back(
                    prim->loops()[start],
                    prim->loops()[start + 1],
                    prim->loops()[start + 2]);
            for (int i = 3; i < len; i++) {
                prim->tris.emplace_back(
                        prim->loops()[start],
                        prim->loops()[start + i - 1],
                        prim->loops()[start + i]);
            }
        }
    }
}

}
