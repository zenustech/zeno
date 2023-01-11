#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
using namespace zeno;

/**
 * @brief 这个节点是最终求解PBD的节点。
 * 
 */
struct PBDPostSolve : zeno::INode {
    void postSolve(const zeno::AttrVector<zeno::vec3f> &pos,
                   const std::vector<zeno::vec3f> &prevPos,
                   std::vector<zeno::vec3f> &vel,
                   std::vector<float> &invMass,
                   float dt)
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            if(invMass[i] == 0)
            {
                continue;
            }
            vel[i] = (pos[i] - prevPos[i]) / dt;
        }
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");

        float dt = prim->userData().getLiterial<float>("dt");

        auto &pos = prim->verts;
        auto &vel = prim->verts.attr<vec3f>("vel");
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");
        auto &invMass = prim->verts.attr<float>("invMass");

        postSolve(pos, prevPos, vel, invMass, dt);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBDPostSolve,
    {
        {{"prim"}},
        {"outPrim"},
        {},
        {"PBD"}
    }
)