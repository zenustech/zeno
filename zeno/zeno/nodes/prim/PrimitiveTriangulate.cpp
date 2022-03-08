#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>

namespace zeno {
namespace {
struct PrimitiveTriangulate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (get_param<bool>("from_poly")) {
            if (get_param<bool>("poly_with_uv")) {
                prim_polys_to_tris_with_uv(prim.get());
            } else {
                prim_polys_to_tris(prim.get());
            }
        }
        if (get_param<bool>("from_quads")) {
            prim_quads_to_tris(prim.get());
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveTriangulate,
        { /* inputs: */ {
        {"primitive", "prim"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        {"bool", "from_poly", "1"},
        {"bool", "poly_with_uv", "1"},
        {"bool", "from_quads", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
