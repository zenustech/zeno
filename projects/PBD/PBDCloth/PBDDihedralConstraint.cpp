#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>

using namespace zeno;
struct PBDDihedralConstraint : zeno::INode {

    void constraint(
        const vec3f & p1,
        const vec3f & p2,
        const vec3f & p3,
        const vec3f & p4,
        const float invMass1,
        const float invMass2,
        const float invMass3,
        const float invMass4,
        const float restAng,
        const float dihedralCompliance,
        const float dt,
        const vec3f &n1,
        const vec3f &n2,
        vec3f & dp1,
        vec3f & dp2,
        vec3f & dp3,
        vec3f & dp4,
        )
    {
        float alpha = dihedralCompliance / dt / dt;

        //compute grads
        p2 = p2 - p1;
        p3 = p3 - p1;
        p4 = p4 - p1;
        float d = dot(n1,n2);
        grad3 =  (cross(p2,n2) + cross(n1,p2) * d) / length(cross(p2,p3));
        grad4 =  (cross(p2,n1) + cross(n2,p2) * d) / length(cross(p2,p4));
        grad2 = -(cross(p3,n2) + cross(n1,p3) * d) / length(cross(p2,p3))
                        -(cross(p4,n1) + cross(n2,p4) * d) / length(cross(p2,p4));
        grad1 = - grad2 - grad3 - grad4;

        //compute denominator
        float w = 0.0;
        w += invMass1 * (length(grad1)) * (length(grad1));
        w += invMass2 * (length(grad2)) * (length(grad2));
        w += invMass3 * (length(grad3)) * (length(grad3));
        w += invMass4 * (length(grad4)) * (length(grad4));
        
        //compute scaling factor
        float ang = acos(d);
        float C = (ang - restAng);
        float s = -C * sqrt(1-d*d) /(w + alpha);
        
        dpos1 = grad1 * s * invMass1;
        dpos2 = grad2 * s * invMass2;
        dpos3 = grad3 * s * invMass3;
        dpos4 = grad4 * s * invMass4;
    }


public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto dihedralCompliance = get_input<zeno::NumericObject>("dihedralCompliance")->get<float>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();

        auto i = get_input<zeno::NumericObject>("triangle1")->get<int>();
        auto j = get_input<zeno::NumericObject>("triangle2")->get<int>();

        //取出数据。 假设 三角形为1-2-4， 2-3-4， 边2-4 为邻接边
        const vec3i &tri1 = prim->tris[i];
        const vec3i &tri2 = prim->tris[j];
        int id1 = tri1[0];
        int id2 = tri1[1];
        int id3 = tri2[1];
        int id4 = tri1[2];
        const auto & pos = prim->verts;
        const vec3f & p1 = pos[id1];
        const vec3f & p2 = pos[id2];
        const vec3f & p3 = pos[id3];
        const vec3f & p4 = pos[id4];
        const auto &invMass = prim->verts.attr<float>("invMass");
        const float invMass1 =  invMass[id1];
        const float invMass2 =  invMass[id2];
        const float invMass3 =  invMass[id3];
        const float invMass4 =  invMass[id4];
        const float restAng = prim->tris.attr<float>("restAng")[];

        constraint(
        p1,
        p2,
        p3,
        p4,
        invMass1,
        invMass2,
        invMass3,
        invMass4,
        restAng,
        dihedralCompliance,
        dt,
        n1,
        n2,
        dp1,
        dp2,
        dp3,
        dp4,
        );


        set_output("outPrim", std::move(prim));
    };
};

ZENDEFNODE(PBDDihedralConstraint, {// inputs:
                 {
                    {"PrimitiveObject", "prim"},
                    {"float", "dihedralCompliance", "0.0"},
                    {"float", "dt", "0.0016667"},
                    {"int", "triangle1", ""},
                    {"int", "triangle2", ""},
                },
                 // outputs:
                 {"outPrim"},
                 // params:
                 {},
                 //category
                 {"PBD"}});