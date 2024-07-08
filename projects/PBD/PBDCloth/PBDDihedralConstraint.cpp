#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>

using namespace zeno;
struct PBDDihedralConstraint : zeno::INode {
    inline vec3f calcNormal(const vec3f & vec1, const vec3f & vec2)
    {
        auto res = cross(vec1, vec2);
        res = res / length(res);
        return res;
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
        const float  restAng,
        float dihedralCompliance,
        float dt,
        std::array<vec3f,4> & dpos
        )
    {
        float alpha = dihedralCompliance / dt / dt;

        vec3f grad[4] = {vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0), vec3f(0,0,0)};

        //计算梯度。先对每个点求相对p1的位置。只是为了准备数据。
        vec3f p1 = pos[0];
        vec3f p2 = pos[1] - p1;
        vec3f p3 = pos[2] - p1;
        vec3f p4 = pos[3] - p1;
        vec3f n1 = calcNormal(p2, p3); //p2与p3叉乘所得面法向
        vec3f n2 = calcNormal(p2, p4);
        float d = dot(n1,n2); 

        //参考Muller2006附录公式(25)-(28)
        grad[2] =  (cross(p2,n2) + cross(n1,p2) * d) / length(cross(p2,p3));
        grad[3] =  (cross(p2,n1) + cross(n2,p2) * d) / length(cross(p2,p4));
        grad[1] = -(cross(p3,n2) + cross(n1,p3) * d) / length(cross(p2,p3))
                        -(cross(p4,n1) + cross(n2,p4) * d) / length(cross(p2,p4));
        grad[0] = - grad[1] - grad[2] - grad[3];

        //公式（8）的分母
        float w = 0.0;
        for (int j = 0; j < 4; j++)
            w += invMass[j] * (length(grad[j])) * (length(grad[j]));
        if(w==0.0)
            return; //防止分母为0
        
        //公式（8）。sqrt(1-d*d)来源请看公式（29），实际上来自grad。
        float ang = acos(d);
        float C = (ang - restAng);
        float s = -C * sqrt(1-d*d) /(w + alpha);

        for (int j = 0; j < 4; j++)
            dpos[j] = grad[j] * s * invMass[j];
    }


public:
    virtual void apply() override {
        auto dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();

        auto p1 = get_input<zeno::NumericObject>("p1")->get<zeno::vec3f>();
        auto p2 = get_input<zeno::NumericObject>("p2")->get<zeno::vec3f>();
        auto p3 = get_input<zeno::NumericObject>("p3")->get<zeno::vec3f>();
        auto p4 = get_input<zeno::NumericObject>("p4")->get<zeno::vec3f>();
        auto invMass1 = get_input<zeno::NumericObject>("invMass1")->get<float>();
        auto invMass2 = get_input<zeno::NumericObject>("invMass2")->get<float>();
        auto invMass3 = get_input<zeno::NumericObject>("invMass3")->get<float>();
        auto invMass4 = get_input<zeno::NumericObject>("invMass4")->get<float>();
        auto restAng4p = get_input<zeno::NumericObject>("restAng4p")->get<float>();

        std::array<vec3f,4> pos4p{p1,p2,p3,p4};
        std::array<vec3f,4> dpos4p;
        for (auto & x: dpos4p)
            x = vec3f{0,0,0};
        std::array<float,4> invMass4p{invMass1, invMass2, invMass3, invMass4};

        dihedralConstraint(pos4p, invMass4p, restAng4p, dihedralCompliance, dt,  dpos4p);

        auto  dpos1 = std::make_shared<NumericObject> (dpos4p[0]);
        auto  dpos2 = std::make_shared<NumericObject> (dpos4p[1]);
        auto  dpos3 = std::make_shared<NumericObject> (dpos4p[2]);
        auto  dpos4 = std::make_shared<NumericObject> (dpos4p[3]);

        set_output("dpos1", std::move(dpos1));
        set_output("dpos2", std::move(dpos2));
        set_output("dpos3", std::move(dpos3));
        set_output("dpos4", std::move(dpos4));
    };
};

ZENDEFNODE(PBDDihedralConstraint, {// inputs:
                 {
                    {"float", "dihedralCompliance", "0.0"},
                    {"float", "dt", "0.0016667"},
                    {"vec3f", "p1", ""},
                    {"vec3f", "p2", ""},
                    {"vec3f", "p3", ""},
                    {"vec3f", "p4", ""},
                    {"float", "invMass1", ""},
                    {"float", "invMass2", ""},
                    {"float", "invMass3", ""},
                    {"float", "invMass4", ""},
                    {"float", "restAng4p", ""},
                },
                 // outputs:
                 {
                    {"vec3f", "dpos1"},
                    {"vec3f", "dpos2"},
                    {"vec3f", "dpos3"},
                    {"vec3f", "dpos4"},
                 },
                 // params:
                 {},
                 //category
                 {"PBD"}});
