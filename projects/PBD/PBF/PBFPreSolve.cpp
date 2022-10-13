#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
using namespace zeno;

/**
 * @brief 这个节点仅仅做测试用，后续会和PBDPreSolve合并。
 * 
 */
struct PBFPreSolve : zeno::INode {
    vec3f bounds;
    float pRadius;
    float worldScale;

    void preSolve(  std::vector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    std::vector<zeno::vec3f> &vel,
                    vec3f & externForce,
                    float dt)
    {
        for (int i = 0; i < pos.size(); i++)
            prevPos[i] = pos[i];

        //update the pos
        for (int i = 0; i < pos.size(); i++)
        {
            vec3f tempVel = vel[i];
            tempVel += externForce * dt;
            pos[i] += tempVel * dt;
            boundaryHandling(pos[i]);
        }

        // neighborSearch();//grid-baed neighborSearch for now
    }

    void boundaryHandling(vec3f & p)
    {
        float bmin = pRadius/worldScale;
        vec3f bmax = bounds - pRadius/worldScale;

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
        auto prim = get_input<PrimitiveObject>("prim");

        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        prim->userData().set("dt", std::make_shared<NumericObject>(dt));

        auto externForce = get_input<zeno::NumericObject>("externForce")->get<vec3f>();

        auto &pos = prim->verts;

        if(!prim->has_attr("vel"))
            prim->verts.add_attr<vec3f>("vel");
        if(!prim->has_attr("prevPos"))
            prim->verts.add_attr<vec3f>("prevPos");

        auto &vel = prim->verts.attr<vec3f>("vel");
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");

        //get parameters
        bounds = get_input<zeno::NumericObject>("bounds")->get<vec3f>();
        pRadius = get_input<zeno::NumericObject>("pRadius")->get<float>();
        worldScale = get_input<zeno::NumericObject>("worldScale")->get<float>();

        // std::fill(invMass.begin(),invMss.end(),1.0/mass)

        preSolve(pos, prevPos, vel, externForce, dt);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBFPreSolve,
    {
        {
            {"prim"},
            {"float","dt","0.05"},
            {"float","pRadius","3.0"},
            {"vec3f","bounds","40.0, 40.0, 40.0"},
            {"float","worldScale","20.0"},
            {"vec3f", "externForce", "0.0, -10.0, 0.0"}
        },
        {"outPrim"},
        {},
        {"PBD"}
    }
)