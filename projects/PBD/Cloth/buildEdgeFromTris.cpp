#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

/**
 * @brief 从三角形重建边
 * 
 */
struct buildEdgeFromTris : zeno::INode {
    void build(const AttrVector<vec3i> &tris, std::vector<vec2i> &edge)
    {
        struct myLess {
            bool operator()(vec2i const &a, vec2i const &b) const {
                return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1])) <
                       std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
            }
        };
        std::set<vec2i, myLess> ee;
        for(int i= 0; i<tris.size(); i++)
        {
            int q0 = tris[i][0];
            int q1 = tris[i][1];
            int q2 = tris[i][2];

            auto e1 = vec2i(q0,q1);
            auto e2 = vec2i(q1,q2);
            auto e3 = vec2i(q2,q0);

            //使用std::set去重复边
            ee.emplace(e1);
            ee.emplace(e2);
            ee.emplace(e3);
        }
        edge.resize(ee.size());
        edge.assign(ee.begin(), ee.end());
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &tris = prim->tris;
        auto &edge = prim->lines;

        build(tris,edge);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    buildEdgeFromTris,{
        {"prim"},
        {"outPrim"},
        {},
        {"PBD"}
    }
)