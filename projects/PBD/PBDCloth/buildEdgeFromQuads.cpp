#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
using namespace zeno;

/**
 * @brief 用来重建布料的边连接关系。（实际上 primWireFrame 节点已经达成了相同功能，
 * 可以使用那个节点代替这个）
 * 
 */
struct buildEdgeFromQuads : zeno::INode {
    void build(const AttrVector<vec4i> &quads, std::vector<vec2i> &edge)
    {
        struct myLess {
            bool operator()(vec2i const &a, vec2i const &b) const {
                return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1])) <
                       std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
            }
        };
        std::set<vec2i, myLess> ee;
        for(int i= 0; i<quads.size(); i++)
        {
            int q0 = quads[i][0];
            int q1 = quads[i][1];
            int q2 = quads[i][2];
            int q3 = quads[i][3];

            auto e1 = vec2i(q0,q1);
            auto e2 = vec2i(q1,q3);
            auto e3 = vec2i(q3,q0);
            auto e4 = vec2i(q1,q2);
            auto e5 = vec2i(q3,q2);

            //使用std::set去重复边
            ee.emplace(e1);
            ee.emplace(e2);
            ee.emplace(e3);
            ee.emplace(e4);
            ee.emplace(e5);
        }
        edge.resize(ee.size());
        edge.assign(ee.begin(), ee.end());
    }

    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &quads = prim->quads;
        auto &edge = prim->lines;

        build(quads,edge);

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(
    buildEdgeFromQuads,{
        {gParamType_Primitive, "prim"},
        {"outPrim"},
        {},
        {"PBD"}
    }
)