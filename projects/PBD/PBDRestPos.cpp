#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <iostream>

namespace zeno {
struct PBDRestPos : INode
{
public:
        virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->verts;
        auto &restPos = prim->verts.add_attr<vec3f>("restPos");

        std::copy(pos.begin(),pos.end(),restPos.begin());

        set_output("outPrim",std::move(prim));
    };
};

ZENDEFNODE(PBDRestPos, {// inputs:
                 {gParamType_Primitive, "prim"},
                 // outputs:
                {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});
} // namespace zeno


