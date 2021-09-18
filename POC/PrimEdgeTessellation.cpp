#if 0
#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace {
using namespace zeno;

struct PrimEdgeTessellation : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        for (int i = 0; i < prim->lines.size(); i++) {
            auto line = prim->lines[i];
            auto v0 = prim->verts[line[0]];
            auto v1 = prim->verts[line[1]];
        } // TODO

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimEdgeTessellation)({
        {"prim"},
        {"prim"},
        {},
        {"cgmesh"},
});

}
#endif
