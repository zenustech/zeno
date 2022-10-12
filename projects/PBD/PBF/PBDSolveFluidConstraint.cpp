#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct PBDSolveFluidConstraint : zeno::INode {
private:

    void solve(PrimitiveObject *prim)
    {

    }


public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");

        solve(prim.get());

        //output
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveFluidConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "rest_density", "1000.0"},
                    {"float", "dt", "0.0016667"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno