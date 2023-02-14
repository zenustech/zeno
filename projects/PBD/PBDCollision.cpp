#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

/**
 * @brief 求解碰撞。WIP。尚未完成。
 * 
 */
struct PBDCollision : zeno::INode {
    void preSolve(  zeno::AttrVector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    std::vector<zeno::vec3f> &vel,
                    float dt
                    )
    {
        vec3f center {0,0.2,0};
        float radius = 0.1;


        for (int i = 0; i < pos.size(); i++) 
        {
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }

            //ball
            if (length(pos[i] - center) < radius) 
            {
                vec3f normal = normalize(pos[i] - center);
                vec3f v = normalize(min(dot(pos[i],normal),0)*normal);
                vel[i] -= v;
                pos[i] +=vel[i] * dt;
            }
        }
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto dt = get_input<NumericObject>("dt")->get<float>();

        auto &pos = prim->verts;
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");
        auto &vel = prim->verts.attr<vec3f>("vel");

        preSolve(pos, prevPos,vel,dt);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    PBDCollision,
    {
        {
            {"prim"}
        },
        {"outPrim"},
        {},
        {"PBD"}
    }
)