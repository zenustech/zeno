#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
namespace zeno {
struct PBDCloth : zeno::INode {
private:
    //physical param
    zeno::vec3f externForce{0, -10.0, 0};
    int numSubsteps = 10;
    float dt = 1.0 / 60.0 / numSubsteps;
    float edgeCompliance = 100.0;
    float dihedralCompliance = 1.0;

    std::vector<float> restLen;
    std::vector<float> restVol;
    std::vector<float> invMass;
    std::vector<float> restAng;

    std::vector<zeno::vec3f> prevPos;
    std::vector<zeno::vec3f> vel;

    int numParticles;

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


    void preSolve(  zeno::AttrVector<zeno::vec3f> &pos,
                    std::vector<zeno::vec3f> &prevPos,
                    std::vector<zeno::vec3f> &vel)
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

    /**
     * @brief 求解PBD所有边约束（也叫距离约束）。目前采用Gauss-Seidel方式（难以并行）。
     * 
     * @param pos 点位置
     * @param edge 边连接关系
     * @param invMass 点质量的倒数
     * @param restLen 边的原长
     * @param edgeCompliance 柔度（越小约束越强，最小为0）
     * @param dt 时间步长
     */
    void solveDistanceConstraint( 
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec2i> &edge,
        const std::vector<float> & invMass,
        const std::vector<float> & restLen,
        const float edgeCompliance,
        const float dt
        )
    {
        float alpha = edgeCompliance / dt / dt;
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
        const zeno::AttrVector<float> &invMass,
        const zeno::AttrVector<float> &restAng,
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

    void postSolve(const zeno::AttrVector<zeno::vec3f> &pos,
                   const std::vector<zeno::vec3f> &prevPos,
                   std::vector<zeno::vec3f> &vel)
    {
        for (int i = 0; i < pos.size(); i++) 
            vel[i] = (pos[i] - prevPos[i]) / dt;
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();

        numSubsteps = get_input<zeno::NumericObject>("numSubsteps")->get<int>();
        edgeCompliance = get_input<zeno::NumericObject>("edgeCompliance")->get<float>();
        dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();

        dt = 1.0/60.0/numSubsteps;
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        auto &quads = prim->quads;
        auto &surf = prim->tris;

        for (size_t i = 0; i < 1; i++)
        {
            for (int steps = 0; steps < numSubsteps; steps++) 
            {
                preSolve(pos, prevPos, vel);
                solveDistanceConstraint(pos, edge, invMass, restLen ,edgeCompliance, dt);
                solveDihedralConstraint(pos, quads, invMass, restAng, dihedralCompliance, dt);
                postSolve(pos, prevPos, vel);
            }
        }

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDCloth, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"vec3f", "externForce", "0.0, -10.0, 0.0"},
                    {"int", "numSubsteps", "10"},
                    {"float", "edgeCompliance", "100.0"},
                    {"float", "dihedralCompliance", "100.0"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno