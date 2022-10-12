#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../Utils/myPrint.h"
#include <zeno/types/UserData.h>

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
     * @brief 对所有的点求解二面角约束
     * 
     * @param prim 所传入的所有数据
     */
    void solve(PrimitiveObject * prim)
    {
        auto &tris = prim->tris;
        auto &pos = prim->verts;
        auto &adj4th = prim->tris.attr<vec3i>("adj4th");
        if(!prim->has_attr("dpos"))
            prim->add_attr<zeno::vec3f>("dpos");
        auto &dpos = prim->verts.attr<vec3f>("dpos");
        auto &restAng = prim->tris.attr<vec3f>("restAng");
        auto &invMass = prim->verts.attr<float>("invMass");
        float dihedralCompliance = prim->userData().getLiterial<float>("dihedralCompliance");
        float dt = prim->userData().getLiterial<float>("dt");
        float isGaussSidel = prim->userData().getLiterial<bool>("isGaussSidel");

        for (int i = 0; i < tris.size(); i++) //对所有三角面
        {
            for(int k=0; k<3; k++) //三个边，对应着三个邻接面
            {
                int id4 = adj4th[i][k]; //取出第四个点编号
                if (id4 == -1) //如果编号为-1，证明没有这个邻接面
                    continue;

                //对四个点进行求解。注意顺序要按照Muller2006论文中的Fig4。1-2是共享边。3是自己的点，4是对方的点。
                int id1 = tris[i][0];
                int id2 = tris[i][1];
                int id3 = tris[i][2];
                vec4i id{id1,id2,id3,id4};

                vec4f invMass4p{invMass[id[0]],invMass[id[1]],invMass[id[2]],invMass[id[3]]}; //4个点的invMass
                float restAng4p{restAng[i][k]}; // 四个点的原角度
                std::array<vec3f,4>  pos4p{pos[id[0]],pos[id[1]],pos[id[2]],pos[id[3]]}; 
                std::array<vec3f,4>  dpos4p{vec3f{0.0,0.0,0.0},vec3f{0.0,0.0,0.0},vec3f{0.0,0.0,0.0},vec3f{0.0,0.0,0.0}}; //四个点的dpos，也就是待求解的对pos的修正值。

                //这里只传入需要的四个点的数据，求解得到4个dpos
                dihedralConstraint(pos4p, invMass4p, restAng4p, dihedralCompliance, dt,  dpos4p);

                for (size_t j = 0; j < 4; j++)
                    dpos[id[j]] = dpos4p[j];

                if (isGaussSidel) //高斯赛德尔法在原地修正pos
                    for (size_t j = 0; j < 4; j++)
                        pos[id[j]] += dpos[j];
            }
        }
    }


    /**
     * @brief 求解给定四个点二面角约束。用于布料。
     * 
     * @param pos 输入的四个点位置
     * @param invMass 输入四个点质量倒数
     * @param restAng 输入四个点之间的夹角
     * @param dihedralCompliance 物理参数，柔度
     * @param dt 物理参数，时间步长
     * @param dpos 待求解的对4个位置的修正值。是返回值。
     */
    void dihedralConstraint(
        const std::array<vec3f,4> & pos,
        const vec4f & invMass,
        const float restAng,
        const float dihedralCompliance,
        const float dt,
        std::array<vec3f,4> & dpos
        )
    {
        if(restAng==std::numeric_limits<float>::lowest())
            return;
        
        float alpha = dihedralCompliance / dt / dt;

        vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};

        //计算梯度。先对每个点求相对p1的位置。只是为了准备数据。
        const vec3f& p1 = pos[0];
        const vec3f& p2 = pos[1] - p1;
        const vec3f& p3 = pos[2] - p1;
        const vec3f& p4 = pos[3] - p1;
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
            w += invMass[j] * (length(grad[j])) * (length(grad[j]));
        if(w==0.0)
            return; //防止分母为0
        
        //公式(8)。sqrt(1-d*d)来源请看公式(29)，实际上来自grad。
        float ang = acos(d);
        float C = (ang - restAng);
        float s = -C * sqrt(1-d*d) /(w + alpha);

        for (int j = 0; j < 4; j++)
            dpos[j] = grad[j] * s * invMass[j];
    }
    


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        //物理参数
        auto dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();
        auto isGaussSidel = get_input<zeno::NumericObject>("isGaussSidel")->get<bool>();
        prim->userData().set("isGaussSidel", std::make_shared<NumericObject>((bool)isGaussSidel));
        prim->userData().set("dihedralCompliance", std::make_shared<NumericObject>((float)dihedralCompliance));
        
        auto dt = prim->userData().getLiterial<float>("dt");
        
        //求解
        solve(prim.get());

        //传出数据
        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDSolveDihedralConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dihedralCompliance", "0.0"},
                    {"bool", "isGaussSidel", "1"},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});