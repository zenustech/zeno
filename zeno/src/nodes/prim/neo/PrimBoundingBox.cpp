#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject *prim) {
    if (!prim->verts.size())
        return {{0, 0, 0}, {0, 0, 0}};
    return parallel_reduce_minmax(prim->verts.begin(), prim->verts.end());
}

namespace {

struct PrimBoundingBox : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto [bmin, bmax] = primBoundingBox(prim.get());
    set_output2("bmin", bmin);
    set_output2("bmax", bmax);
  }
};

ZENDEFNODE(PrimBoundingBox, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"vec3f", "bmin"},
    {"vec3f", "bmax"},
    },
    {
    },
    {"primitive"},
});

}
}
