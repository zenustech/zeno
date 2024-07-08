#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
using namespace zeno;

/**
 * @brief 这个节点是第一个求解PBD的节点。
 * 
 */
struct PBDPreSolve : zeno::INode {
    void preSolve(  zeno::AttrVector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    std::vector<zeno::vec3f> &vel,
                    vec3f & externForce,
                    std::vector<float> &invMass,
                    float dt)
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            if(invMass[i]==0.0)
                continue;
            prevPos[i] = pos[i];
            vel[i] += (externForce) * dt;
            pos[i] += vel[i] * dt;
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");

        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        prim->userData().set("dt", std::make_shared<NumericObject>(dt));

        auto externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();

        auto &pos = prim->verts;
        auto &invMass = prim->verts.attr<float>("invMass");

        if(!prim->has_attr("vel"))
            prim->verts.add_attr<vec3f>("vel");
        if(!prim->has_attr("prevPos"))
            prim->verts.add_attr<vec3f>("prevPos");

        auto &vel = prim->verts.attr<vec3f>("vel");
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");

        preSolve(pos, prevPos, vel, externForce, invMass, dt);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBDPreSolve,
    {
        {
            {"prim"},
            {"float","dt","0.0016667"},
            {"vec3f", "externForce", "0.0, -10.0, 0.0"}
        },
        {"outPrim"},
        {},
        {"PBD"}
    }
)