#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/orthonormal.h>

namespace zeno {

struct PrimitiveScale : zeno::INode {
    virtual void apply() override {
        auto origin = get_input<zeno::NumericObject>("origin")->get<zeno::vec3f>();
        auto axis = get_input<zeno::NumericObject>("axis")->get<zeno::vec3f>();
        auto scale = get_input<zeno::NumericObject>("scale")->get<float>();
        auto prim = get_input<PrimitiveObject>("prim");
        #pragma omp parallel for
        for (int i = 0; i < prim->verts.size(); i++) {
            auto pos = prim->verts[i];
            pos -= origin;
            pos += scale * dot(pos, axis) * axis;
            pos += origin;
            prim->verts[i] = pos;
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveScale, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "origin", "0,0,0"},
    {"vec3f", "axis", "0,1,0"},
    {"float", "scale", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {},
    {"primitive"},
});

}
