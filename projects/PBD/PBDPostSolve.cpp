#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

/**
 * @brief 这个节点是最终求解PBD的节点。
 * 
 */
struct PBDPostSolve : zeno::INode {
    void postSolve(const zeno::AttrVector<zeno::vec3f> &pos,
                   const std::vector<zeno::vec3f> &prevPos,
                   std::vector<zeno::vec3f> &vel,
                   float dt)
    {
        for (int i = 0; i < pos.size(); i++) 
            vel[i] = (pos[i] - prevPos[i]) / dt;
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");

        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();

        auto &pos = prim->verts;
        auto &vel = prim->verts.attr<vec3f>("vel");
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");

        postSolve(pos, prevPos, vel, dt);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBDPostSolve,
    {
        {{"prim"},{"float","dt","0.0016667"}},
        {"outPrim"},
        {},
        {"PBD"}
    }
)