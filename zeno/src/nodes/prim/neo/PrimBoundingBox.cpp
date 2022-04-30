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
    return {{0, 0, 0}, {0, 0, 0}};
    std::transform_reduce(ZENO_PAR_UNSEQ prim->verts.begin(), prim->verts.end()
            , /*identity=*/std::make_pair(*prim->verts.begin(), *prim->verts.begin())
            , /*reduce=*/[] (auto const &l, auto const &r) {
                return std::make_pair(zeno::min(l.first, r.first), zeno::max(l.second, r.second));
            }
            , /*transform=*/[] (auto const &l) {
                return std::make_pair(l, l);
            }
            );
    /* return {parallel_reduce_min(par_unseq, prim->verts.begin(), prim->verts.end()), */
    /*         parallel_reduce_max(par_unseq, prim->verts.begin(), prim->verts.end())}; */
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
