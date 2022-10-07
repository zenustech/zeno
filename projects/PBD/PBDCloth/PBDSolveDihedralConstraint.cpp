#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../Utils/myPrint.h"

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

    inline vec3f calcNormal(const vec3f & vec1, const vec3f & vec2)
    {
        auto res = cross(vec1, vec2);
        res = res / length(res);
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
        PrimitiveObject * prim,
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec4i> &quads,
        const std::vector<float> &invMass,
        const std::vector<float> &restAng,
        const float dihedralCompliance,
        const float dt
        )
    {
        auto & grad1 = prim->quads.add_attr<vec3f>("grad1");
        auto & grad2 = prim->quads.add_attr<vec3f>("grad2");
        auto & grad3 = prim->quads.add_attr<vec3f>("grad3");
        auto & grad4 = prim->quads.add_attr<vec3f>("grad4");
        auto & s1 = prim->quads.add_attr<float>("s");
        auto & invMass1 = prim->quads.add_attr<vec4f>("invMass1");
        auto & angle1 = prim->quads.add_attr<float>("angle");
        auto & id1 = prim->quads.add_attr<vec4i>("id");
        auto & n11 = prim->quads.add_attr<vec3f>("n1");
        auto & n21 = prim->quads.add_attr<vec3f>("n2");
        auto & d1 = prim->quads.add_attr<float>("d1");
        auto & dpos1 = prim->quads.add_attr<vec3f>("dpos1");
        auto & dpos2 = prim->quads.add_attr<vec3f>("dpos2");
        auto & dpos3 = prim->quads.add_attr<vec3f>("dpos3");
        auto & dpos4 = prim->quads.add_attr<vec3f>("dpos4");

        float alpha = dihedralCompliance / dt / dt;
        vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};
        vec4i id{-1,-1,-1,-1};
        vec3f dpos[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};
 
        for (int i = 0; i < quads.size(); i++)
        {
            for (int j = 0; j < 4; j++)
                id[j] = quads[i][j];

            id1[i] = id;

            //compute grads
            vec3f p1 = pos[id[0]];
            vec3f p2 = pos[id[1]] - p1;
            vec3f p3 = pos[id[2]] - p1;
            vec3f p4 = pos[id[3]] - p1;
            vec3f n1 = calcNormal(p2, p3);
            vec3f n2 = calcNormal(p2, p4);
            float d = dot(n1,n2);

            n11[i]=n1;
            n21[i]=n2;
            d1[i]=d;

            grad[2] =  (cross(p2,n2) + cross(n1,p2) * d) / length(cross(p2,p3));
            grad[3] =  (cross(p2,n1) + cross(n2,p2) * d) / length(cross(p2,p4));
            grad[1] = -(cross(p3,n2) + cross(n1,p3) * d) / length(cross(p2,p3))
                          -(cross(p4,n1) + cross(n2,p4) * d) / length(cross(p2,p4));
            grad[0] = - grad[1] - grad[2] - grad[3];

            grad1[i] = grad[0];
            grad2[i] = grad[1];
            grad3[i] = grad[2];
            grad4[i] = grad[3];

            //compute denominator
            float w = 0.0;
            for (int j = 0; j < 4; j++)
                w += invMass[id[j]] * (length(grad[j])) * (length(grad[j])) ;
            
            if(w==0.0)
                continue;
            //compute scaling factor
            float ang = acos(d);
            float C = (ang - restAng[i]);
            float s = -C * sqrt(1-d*d) /(w + alpha);
            
            angle1[i] = ang;
            s1[i] = s;

            for (int j = 0; j < 4; j++)
            {
                dpos[j] = grad[j] * s * invMass[id[j]];
                pos[id[j]] += grad[j] * s * invMass[id[j]];
            }

            dpos1[i] = dpos[0];
            dpos2[i] = dpos[1];
            dpos3[i] = dpos[2];
            dpos4[i] = dpos[3];
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

        solveDihedralConstraint(prim.get(),pos, quads, invMass, restAng, dihedralCompliance, dt);

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveDihedralConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dihedralCompliance", "0.0"},
                    {"float", "dt", "0.0016667"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});