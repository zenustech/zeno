#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <algorithm>
#include <iostream>
#include "../Utils/myPrint.h"
namespace zeno {
struct PBDCloth : zeno::INode {
private:
    //physical param
    zeno::vec3f externForce{0, -10.0, 0};
    int numSubsteps = 15;
    float dt = 1.0 / 60.0 / numSubsteps;
    float edgeCompliance = 0.0;
    float dihedralCompliance;

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
     * @brief 计算球SDF(测试用)
     * 
     * @param p 当前点的位置
     * @param normal 返回的sdf的梯度（归一化之后）
     * @return float 返回的sdf值
     */
    float ballSdf(const vec3f& p, vec3f& normal)
    {
        vec3f ballCenter{0, 0.5, 0};
        float ballRadius{0.1};

        const vec3f diff = p-ballCenter;
        float dist = length(diff) - ballRadius;
        normal = normalize(diff);
        return dist;
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

            //地板碰撞
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }

            //球体SDF碰撞
            vec3f normal;
            float sdf = ballSdf(pos[i], normal) ;
            auto loss=0.05;
            if(sdf < 0.0)
            {
                pos[i] = pos[i] - sdf * normal;
                if(dot(vel[i],normal)<0.0)
                    vel[i] -= min(dot(vel[i],normal), 0) * normal * loss;
            }
        }
    }

    inline vec3f calcNormal(const vec3f & vec1, const vec3f & vec2)
    {
        auto res = cross(vec1, vec2);
        res = res / length(res);
        return res;
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
              std::vector<vec3f> &pos,
        const std::vector<vec2i> &edge,
        const std::vector<float> &invMass,
        const std::vector<float> &restLen,
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

    void solveStretchingConstraint(
        zeno::AttrVector<zeno::vec3f> &pos,
        const zeno::AttrVector<zeno::vec2i> &edge,
        const std::vector<float> & invMass,
        const std::vector<float> & restLen,
        const float edgeCompliance,
        const float dt
    ) 
    {
        float alpha = edgeCompliance / dt / dt;

        for (auto i = 0; i < edge.size(); i++) {
            int id0 = edge[i][0];
            int id1 = edge[i][1];

            auto w0 = invMass[id0];
            auto w1 = invMass[id1];
            auto w = w0 + w1;
            if (w == 0.0)
                continue;

            auto grads = pos[id0] - pos[id1];
            float Len = length(grads);
            if (Len == 0.0)
                continue;
            grads /= Len;
            auto C = Len - restLen[i];
            auto s = -C / (w + alpha);

            pos[id0] += grads *   s * invMass[id0];
            pos[id1] += grads * (-s * invMass[id1]);
        }
    }


    // void solveBendingDistanceConstraint(
    //           std::vector<vec3f> &pos,
    //     const std::vector<vec2i> &quads,
    //     const std::vector<float> &invMass,
    //     const std::vector<float> &restLen,
    //     const float edgeCompliance,
    //     const float dt
    // )
    // {
    //     auto alpha = compliance / dt /dt;

    //     for (auto i = 0; i < quads.size(); i++) 
    //     {
    //         int id0 = quads[i][2];
    //         int id1 = quads[i][3];

    //         auto w0 = invMass[id0];
    //         auto w1 = invMass[id1];
    //         auto w = w0 + w1;
    //         if (w == 0.0)
    //             continue;

    //         auto grads = pos[id0] - pos[id1];
    //         float Len = length(grads);
    //         if (Len == 0.0)
    //             continue;
    //         grads /= Len;
    //         auto C = Len - restLen[i];
    //         auto s = -C / (w + alpha);
    //         pos[id0] += grads *   s * invMass[id0];
    //         pos[id1] += grads * (-s * invMass[id1]);
    //     }
    // }

    void solveDihedralConstraint(PrimitiveObject *prim)
    {
        vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};
        float alpha = dihedralCompliance / dt / dt;

        auto &tris=prim->tris;
        auto &pos=prim->verts;
        auto &adj4th=prim->tris.attr<vec3i>("adj4th");
        auto &invMass=prim->verts.attr<float>("invMass");
        auto &restAng=prim->tris.attr<vec3f>("restAng");

        for (int i = 0; i < prim->tris.size(); i++)
        {
            for (int j = 0; j < 3; j++) //三个邻接三角面
            {
                const int id1=tris[i][0];
                const int id2=tris[i][1];
                const int id3=tris[i][2];
                const int id4=adj4th[i][j];
                const vec4f invMass_{invMass[id1],invMass[id2],invMass[id3],invMass[id4]};

                //计算梯度。先对每个点求相对p1的位置。只是为了准备数据。
                const vec3f& p1 = pos[id1];
                const vec3f& p2 = pos[id2] - p1;
                const vec3f& p3 = pos[id3] - p1;
                const vec3f& p4 = pos[id4] - p1;
                const vec3f& n1 = calcNormal(p2, p3); //p2与p3叉乘所得面法向
                const vec3f& n2 = calcNormal(p2, p4);
                float d = dot(n1,n2); 

                //参考Muller2006附录公式(25)-(28)
                grad[2] =  (cross(p2,n2) + cross(n1,p2) * d) / length(cross(p2,p3));
                grad[3] =  (cross(p2,n1) + cross(n2,p2) * d) / length(cross(p2,p4));
                grad[1] = -(cross(p3,n2) + cross(n1,p3) * d) / length(cross(p2,p3))
                                -(cross(p4,n1) + cross(n2,p4) * d) / length(cross(p2,p4));
                grad[0] = - grad[1] - grad[2] - grad[3];

                //公式(8)的分母
                float w = 0.0;
                for (int j = 0; j < 4; j++)
                    w += invMass_[j] * (length(grad[j])) * (length(grad[j]));
                if(w==0.0)
                    return; //防止分母为0
                
                //公式(8)。sqrt(1-d*d)来源请看公式(29)，实际上来自grad。
                float ang = acos(d);
                float C = (ang - restAng[i][j]);
                float s = -C * sqrt(1-d*d) /(w + alpha);

                for (int j = 0; j < 4; j++)
                    pos[j] += grad[j] * s * invMass[j];
            }
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
        auto &edge = prim->edges;
        auto &quads = prim->quads;
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");
        auto &vel = prim->verts.attr<vec3f>("vel");
        auto &invMass=prim->verts.attr<float>("invMass");
        auto &restLen=prim->edges.attr<float>("restLen");

        for (int steps = 0; steps < numSubsteps; steps++) 
        {
            preSolve(pos, prevPos, vel);
            solveDistanceConstraint(pos, edge, invMass, restLen ,edgeCompliance, dt);
            // solveStretchingConstraint(pos, edge, invMass, restLen ,edgeCompliance, dt);
            // solveDihedralConstraint(prim.get());
            postSolve(pos, prevPos, vel);
        }

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDCloth, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"vec3f", "externForce", "0.0, -10.0, 0.0"},
                    {"int", "numSubsteps", "10"},
                    {"float", "edgeCompliance", "0.0"},
                    {"float", "dihedralCompliance", "1.0"}
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno