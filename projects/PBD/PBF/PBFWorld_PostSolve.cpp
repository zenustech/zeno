#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include "./PBFWorld.h"
using namespace zeno;


struct PBFWorld_PostSolve : zeno::INode {

    void postSolve(PBFWorld *data, PrimitiveObject* prim)
    {
        auto &pos = prim->verts;
        for (size_t i = 0; i < data->numParticles; i++)
            data->vel[i] = (pos[i] - data->prevPos[i]) / data->dt;
    }

     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");

        postSolve(data.get(), prim.get());

        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(PBFWorld_PostSolve,
    {
        {"prim","PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"}
    }
);