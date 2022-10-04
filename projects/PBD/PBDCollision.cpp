#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

/**
 * @brief 求解碰撞。WIP。尚未完成。
 * 
 */
struct PBDCollision : zeno::INode {
    void preSolve(  zeno::AttrVector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos
                    )
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }
    }


     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");

        auto &pos = prim->verts;
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");

        preSolve(pos, prevPos);

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