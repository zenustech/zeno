#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include "./PBFWorld.h"
using namespace zeno;


struct PBFWorld_PostSolve : zeno::INode {

    void postSolve(PBFWorld *data)
    {
        auto &pos = data->prim->verts;
        for (size_t i = 0; i < data->numParticles; i++)
            data->vel[i] = (pos[i] - data->prevPos[i]) / data->dt;
    }

     virtual void apply() override{
        auto data = get_input<PBFWorld>("PBFWorld");
        postSolve(data.get());
        set_output("PBFWorld", std::move(data));
        set_output("outPrim", std::move(data->prim));
    }
};

ZENDEFNODE(PBFWorld_PostSolve,
    {
        {
            {"PBFWorld"}
        },
        {"PBFWorld","outPrim"},
        {},
        {"PBD"}
    }
);