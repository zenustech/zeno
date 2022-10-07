#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <numeric>
#include <iostream>
namespace zeno {
struct PBDSetFixedPoint : zeno::INode {
private:

public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");
        auto fixedPoint = get_input<NumericObject>("fixedPoint")->get<int>();

        auto & invMass = prim->verts.attr<float>("invMass");
        auto & pos = prim->verts;

        // set fixed
        invMass[fixedPoint] = 0.0;

        // output
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSetFixedPoint, {// inputs:
                {
                    {"prim", {"float","fixedPoint",""}}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno