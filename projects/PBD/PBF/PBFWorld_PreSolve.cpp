#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include "./PBFWorld.h"
#include "../Utils/myPrint.h"//debug
using namespace zeno;


struct PBFWorld_PreSolve : zeno::INode {

    void preSolve(  std::vector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    const std::vector<zeno::vec3f> &vel,
                    const vec3f & externForce,
                    const float dt,
                    const vec3f bounds)
    {
        for (int i = 0; i < pos.size(); i++)
            prevPos[i] = pos[i];

        //update the pos
        for (int i = 0; i < pos.size(); i++)
        {
            vec3f tempVel = vel[i];
            tempVel += externForce * dt;
            pos[i] += tempVel * dt;
            boundaryHandling(pos[i], bounds);
        }

    }

    void boundaryHandling(vec3f & p, const vec3f &bounds)
    {
        float bmin = 0.0;
        const vec3f &bmax = bounds;
        for (size_t dim = 0; dim < 3; dim++)
        {
            float r = ((float) rand() / (RAND_MAX));
            if (p[dim] <= bmin)
                p[dim] = bmin + 1e-5 * r;
            else if (p[dim]>= bmax[dim])
                p[dim] = bmax[dim] - 1e-5 * r;
        }
    }


     virtual void apply() override{
        auto data = get_input<PBFWorld>("PBFWorld");
        auto &pos = data->prim->verts;
        
        preSolve(pos, data->prevPos, data->vel, data->externForce, data->dt, data->bounds);

        set_output("prim", std::move(data->prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_PreSolve,
    {
        {
            {"PBFWorld"}
        },
        {"prim","PBFWorld"},
        {},
        {"PBD"}
    }
);