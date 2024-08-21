#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include "./PBFWorld.h"
#include "../Utils/myPrint.h"//debug
#include "../Utils/readFile.h"//debug
using namespace zeno;

struct PBFWorld_PreSolve : zeno::INode {

    void preSolve(PBFWorld* data, PrimitiveObject * prim)
    {
        auto &pos = prim->verts;
        for (int i = 0; i < pos.size(); i++)
            data->prevPos[i] = pos[i];

        //update the pos
        for (int i = 0; i < pos.size(); i++)
        {
            vec3f tempVel = data->vel[i];
            tempVel += data->externForce * data->dt;
            pos[i] += tempVel * data->dt;
            boundaryHandling(pos[i], data->bounds_min, data->bounds_max);
        }

    }

    void boundaryHandling(vec3f & p, const vec3f &bounds_min, const vec3f &bounds_max)
    {
        for (size_t dim = 0; dim < 3; dim++)
        {
            float r = ((float) rand() / (RAND_MAX));//0-1随机数
            if (p[dim] <= bounds_min[dim])
                p[dim] = bounds_min[dim] + 1e-5 * r;
            else if (p[dim]>= bounds_max[dim])
                p[dim] = bounds_max[dim] - 1e-5 * r;
        }
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");

        preSolve(data.get(), prim.get());

        //debug
        // auto &pos = prim->verts;
        // echoVec(pos[100]);
        // std::cout<<"good1\n";
        // printVectorField("pos_preSolve_PBFWorld.csv",pos);
        //end debug

        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_PreSolve,
    {
        {gParamType_Primitive,"PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"}
    }
);