#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
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
                    float dt)
    {
        for (int i = 0; i < pos.size(); i++) 
        {
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
        auto externForce = get_input<zeno::NumericObject>("externForce")->get<vec3f>();

        auto &pos = prim->verts;
        auto &vel = prim->verts.add_attr<vec3f>("vel");
        auto &prevPos = prim->verts.add_attr<vec3f>("prevPos");

        preSolve(pos, prevPos, vel, externForce, dt);

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