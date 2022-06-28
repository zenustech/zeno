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
    auto extraBound = get_input2<float>("extraBound");
    auto [bmin, bmax] = primBoundingBox(prim.get());
    if (extraBound != 0) {
        bmin -= extraBound;
        bmax += extraBound;
    }
    auto center = (bmin + bmax) / 2;
    auto radius = (bmax - bmin) / 2;
    auto diameter = bmax - bmin;
    set_output2("bmin", bmin);
    set_output2("bmax", bmax);
    set_output2("center", center);
    set_output2("radius", radius);
    set_output2("diameter", diameter);
  }
};

ZENDEFNODE(PrimBoundingBox, {
    {
    {"PrimitiveObject", "prim"},
    {"float", "extraBound", "0"},
    },
    {
    {"vec3f", "bmin"},
    {"vec3f", "bmax"},
    {"vec3f", "center"},
    {"vec3f", "radius"},
    {"vec3f", "diameter"},
    },
    {
    },
    {"primitive"},
});

}
}
