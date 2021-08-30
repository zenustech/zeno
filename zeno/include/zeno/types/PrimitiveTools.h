#pragma once

#include <zeno/types/PrimitiveObject.h>

namespace zeno {

static void prim_triangulate(PrimitiveObject *prim) {
    for (auto [start, len]: prim->polys()) {
    }
}

}
