#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <numeric>
#include <iostream>
namespace zeno {
struct PBDPin : zeno::INode {
private:

public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");
        auto pointToPin = get_input<NumericObject>("pointToPin")->get<int>();

        auto & invMass = prim->verts.attr<float>("invMass");
        auto & pos = prim->verts;

        // set fixed
        invMass[pointToPin] = 0.0;

        // output
        set_output("outPrim", std::move(prim));
    };
};
ZENDEFNODE(PBDPin, {// inputs:
                {
                    {"prim", {"float","pointToPin",""}}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno