#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include <iostream>

namespace zeno {
struct PBDSolveDistanceConstraint : zeno::INode {
private:
    /**
     * @brief 求解PBD所有边约束（也叫距离约束）。目前采用Gauss-Seidel方式（难以并行）。
     * 
     * @param pos 点位置
     * @param edge 边连接关系
     * @param invMass 点质量的倒数
     * @param restLen 边的原长
     * @param disntanceCompliance 柔度（越小约束越强，最小为0）
     * @param dt 时间步长
     */
    void solveDistanceConstraint( 
        PrimitiveObject * prim,
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec2i> &edge,
        const std::vector<float> & invMass,
        const std::vector<float> & restLen,
        const float disntanceCompliance,
        const float dt
        )
    {
        float alpha = disntanceCompliance / dt / dt;
        zeno::vec3f grad{0, 0, 0};
        for (int i = 0; i < edge.size(); i++) 
        {
            int id0 = edge[i][0];
            int id1 = edge[i][1];

            grad = pos[id0] - pos[id1];
            float Len = length(grad);
            grad /= Len;
            float C = Len - restLen[i];
            float w = invMass[id0] + invMass[id1];
            float s = -C / (w + alpha);

            pos[id0] += grad *   s * invMass[id0];
            pos[id1] += grad * (-s * invMass[id1]);
        }
    }


public:
    virtual void apply() override {
        //get data
        auto prim = get_input<PrimitiveObject>("prim");

        auto disntanceCompliance = get_input<zeno::NumericObject>("disntanceCompliance")->get<float>();

        float dt = prim->userData().getLiterial<float>("dt");

        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &restLen = prim->lines.attr<float>("restLen");
        auto &invMass = prim->verts.attr<float>("invMass");

        //solve distance constraint
        solveDistanceConstraint(prim.get(), pos, edge, invMass, restLen, disntanceCompliance, dt);

        //output
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveDistanceConstraint, {// inputs:
                 {
                    {gParamType_Primitive, "prim"},
                    {gParamType_Float, "disntanceCompliance", "100.0"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno
