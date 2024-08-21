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
    float dihedralCompliance = 1.0;
    float bendingCompliance = 1.0;

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

    /**
     * @brief 根据外部约束更新位置。碰撞处理在此步内。
     * 
     * @param invMass 输入：质量倒数
     * @param externForce 输入：外力（如重力）
     * @param dt 参数：时间步长 
     * @param pos 更改：位置
     * @param prevPos 更改：前一步位置
     * @param vel 更改：速度 
     */
    void preSolve(  const std::vector<float> &invMass,
                    const vec3f &externForce,
                    const float dt,
                    std::vector<vec3f> &pos,
                    std::vector<vec3f> &prevPos,
                    std::vector<vec3f> &vel) 
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            if (invMass[i] == 0.0)
                continue;

            vel[i] += (externForce) * dt;
            prevPos[i] = pos[i];
            pos[i] += vel[i] * dt;

            //地板碰撞
            if (pos[i][1] < 0.0) 
            {
                pos[i] = prevPos[i];
                pos[i][1] = 0.0;
            }
        }

    }

    /**
     * @brief 反求速度
     * 
     * @param pos 输入：位置
     * @param prevPos 输入：前一时刻位置
     * @param invMass 输入：质量倒数
     * @param dt 参数：时间步长
     * @param vel 输出：速度
     */
    void postSolve( const std::vector<vec3f> &pos,
                    const std::vector<vec3f> &prevPos,
                    const std::vector<float> &invMass,
                    const float dt,
                    std::vector<vec3f> &vel) 
    {
        for (int i = 0; i < pos.size(); i++) 
        {
            if (invMass[i] == 0.0)
                continue;
            vel[i] = (pos[i] - prevPos[i]) / dt;
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
     * @param edge 边连接关系
     * @param invMass 点质量的倒数
     * @param restLen 边的原长
     * @param edgeCompliance 柔度（越小约束越强，最小为0）
     * @param dt 时间步长
     * @param pos 点位置
     */
    void solveDistanceConstraints( 
        const std::vector<vec2i> &edges,
        const std::vector<float> &invMass,
        const std::vector<float> &restLen,
        const float edgeCompliance,
        const float dt,
        std::vector<vec3f> &pos)
    {
        float alpha = edgeCompliance / dt / dt;
        for (auto i = 0; i < edges.size(); i++) 
        {
            int id0 = edges[i][0];
            int id1 = edges[i][1];

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

    /**
     * @brief 利用对角距离法求解弯折约束。
     * 
     * @param quads 三角形对。其中下标2和3代表对角
     * @param invMass 质量倒数
     * @param bendingRestLen 对角距离原长
     * @param bendingCompliance 参数：柔度
     * @param pos 输出：位置
     */
    void solveBendingDistanceConstraints(
        const std::vector<vec4i> &quads,
        const std::vector<float> &invMass,
        const std::vector<float> &bendingRestLen,
        const float bendingCompliance,
        const float dt,
        std::vector<vec3f> &pos)
    {
        auto alpha = bendingCompliance / dt /dt;

        for (auto i = 0; i < quads.size(); i++) 
        {
            int id0 = quads[i][2];
            int id1 = quads[i][3];

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
            auto C = Len - bendingRestLen[i];
            auto s = -C / (w + alpha);
            pos[id0] += grads *   s * invMass[id0];
            pos[id1] += grads * (-s * invMass[id1]);
        }
    }

    void solveDihedralConstraints(PrimitiveObject *prim)
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


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();
        numSubsteps = get_input<zeno::NumericObject>("numSubsteps")->get<int>();
        edgeCompliance = get_input<zeno::NumericObject>("edgeCompliance")->get<float>();
        bendingCompliance = get_input<zeno::NumericObject>("bendingCompliance")->get<float>();
        // dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();

        dt = 1.0/60.0/numSubsteps;
        auto &pos = prim->verts;
        auto &edges = prim->edges;
        auto &quads = prim->quads;
        auto &invMass=prim->verts.attr<float>("invMass");
        auto &restLen=prim->edges.attr<float>("restLen");
        auto &bendingRestLen=prim->quads.attr<float>("bendingRestLen");
        auto &prevPos = prim->verts.attr<vec3f>("prevPos");
        auto &vel = prim->verts.attr<vec3f>("vel");

        static int frames=0;
        frames+=1;

        for (int steps = 0; steps < numSubsteps; steps++) 
        {
            if(frames==100)
                echo(frames);
            preSolve(invMass,externForce, dt,pos,prevPos, vel);
            solveDistanceConstraints(edges, invMass, restLen ,edgeCompliance, dt, pos);
            solveBendingDistanceConstraints(quads,invMass,bendingRestLen,bendingCompliance,dt,pos);
            postSolve(pos,prevPos,invMass,dt,vel);
        }

        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDCloth, {// inputs:
                 {
                    {gParamType_Primitive, "prim"},
                    {gParamType_Vec3f, "externForce", "0.0, -10.0, 0.0"},
                    {gParamType_Int, "numSubsteps", "15"},
                    {gParamType_Float, "edgeCompliance", "0.0"},
                    {gParamType_Float, "bendingCompliance", "1.0"}
                    // {gParamType_Float, "dihedralCompliance", "1.0"},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});

} // namespace zeno