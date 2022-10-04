#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>

using namespace zeno;
struct PBDSolveDihedralConstraint : zeno::INode {
    float computeAng(   const vec3f & p0,
                        const vec3f & p1,
                        const vec3f & p2,
                        const vec3f & p3)
    {
        return acos(computeFaceNormalDot(p0,p1,p2,p3));
    }

    float computeFaceNormalDot( const vec3f & p0,
                                const vec3f & p1,
                                const vec3f & p2,
                                const vec3f & p3)
    {
        auto n1 = cross((p1 - p0), (p2 - p0));
        n1 = n1 / length(n1);
        auto n2 = cross((p1 - p0), (p3 - p0));
        n2 = n2 / length(n2) ;
        auto res = dot(n1, n2);
        if(res<-1.0) res = -1.0;
        if(res>1.0)  res = 1.0;
        return res;
    }

    /**
     * @brief 求解两面夹角约束。注意需要先求出原角度再用。
     * 
     * @param pos 点位置
     * @param quads 一小块布（由四个点组成的两个三角形）的四个顶点编号
     * @param invMass 点的质量倒数
     * @param restAng 原角度
     * @param dihedralCompliance 二面角柔度
     * @param dt 时间步长
     */
    void solveDihedralConstraint(
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec4i> &quads,
        const std::vector<float> &invMass,
        const std::vector<float> &restAng,
        const float dihedralCompliance,
        const float dt
        )
    {
        float alpha = dihedralCompliance / dt / dt;

        for (int i = 0; i < quads.size(); i++)
        {
            //get data
            const auto &invMass1 = invMass[quads[i][0]];
            const auto &invMass2 = invMass[quads[i][1]];
            const auto &invMass3 = invMass[quads[i][2]];
            const auto &invMass4 = invMass[quads[i][3]];

            vec3f &p1 = pos[quads[i][0]];
            vec3f &p2 = pos[quads[i][1]];
            vec3f &p3 = pos[quads[i][2]];
            vec3f &p4 = pos[quads[i][3]];

            //compute grads
            p2 -= p1;
            p3 -= p1;
            p4 -= p1;
            auto n1 = cross(p2, p3);
            n1 = n1 / length(n1);
            auto n2 = cross(p1, p3);
            n2 = n2 / length(n2);
            float d = computeFaceNormalDot(p1,p2,p3,p4);
            vec3f grad3 =  (cross(p2,n2) + cross(n1,p2) * d) / length(cross(p2,p3));
            vec3f grad4 =  (cross(p2,n1) + cross(n2,p2) * d) / length(cross(p2,p4));
            vec3f grad2 = -(cross(p3,n2) + cross(n1,p3) * d) / length(cross(p2,p3))
                          -(cross(p4,n1) + cross(n2,p4) * d) / length(cross(p2,p4));
            vec3f grad1 = - grad2 - grad3 - grad4;

            //compute denominator
            float denom = 0.0;  
            denom += invMass1 * length(grad1)*length(grad1); 
            denom += invMass2 * length(grad2)*length(grad2); 
            denom += invMass3 * length(grad3)*length(grad3); 
            denom += invMass4 * length(grad4)*length(grad4); 
            
            //compute scaling factor
            float ang = computeAng(p1,p2,p3,p4);
            float C = (ang - restAng[i]);
            float s = -C * sqrt(1-d*d) /(denom + alpha);
            
            p1 += grad1 * s * invMass1;
            p2 += grad2 * s * invMass2;
            p3 += grad3 * s * invMass3;
            p4 += grad4 * s * invMass4;
        }
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto &pos = prim->verts;
        auto &quads = prim->quads;
        auto &invMass = prim->verts.attr<float>("invMass");
        auto &restAng = prim->quads.attr<float>("restAng");

        auto dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();

        solveDihedralConstraint(pos, quads, invMass, restAng, dihedralCompliance, dt);

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveDihedralConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dihedralCompliance", "1.0"},
                    {"float", "dt", "0.0016667"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});