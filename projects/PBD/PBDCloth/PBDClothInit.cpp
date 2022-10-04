#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <Eigen/Eigen>

using Matrix4r = Eigen::Matrix<float, 4, 4>;
namespace zeno {
struct PBDClothInit : zeno::INode {
private:
    std::vector<float> restLen;
    std::vector<float> invMass;
    std::vector<float> restAng;
    std::vector<Matrix4r> stiffMatrix{0.0};

    int numParticles;

    float computeAng(
        const vec3f &p0,
        const vec3f &p1, 
        const vec3f &p2, 
        const vec3f &p3)
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

    void initRestLen(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
    }

    void initRestAng(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &quads = prim->quads;
        for(int i = 0; i < quads.size(); i++)
        {
            vec3f &p1 = pos[quads[i][0]];
            vec3f &p2 = pos[quads[i][1]];
            vec3f &p3 = pos[quads[i][2]];
            vec3f &p4 = pos[quads[i][3]];
            restAng[i] = computeAng(p1,p2,p3,p4);
        }
    }

    void initInvMass(PrimitiveObject *prim, float areaDensity) {
        auto &pos = prim->verts;
        auto &quads = prim->quads;
        for(int i = 0; i < quads.size(); i++)
        {
            float quad_area = dot(pos[quads[i][1]] - pos[quads[i][0]],  pos[quads[i][3]] - pos[quads[i][2]]);
            float pInvMass = 0.0;
            pInvMass = areaDensity * quad_area;
            for (int j = 0; j < 4; j++)
                invMass[quads[i][j]] += pInvMass;
        }
    }

    void init(PrimitiveObject *prim, float areaDensity)
    {
        restLen.resize(prim->lines.size());
        invMass.resize(prim->verts.size());
        stiffMatrix.resize(prim->lines.size());

        initRestLen(prim);
        initRestAng(prim);
        initInvMass(prim, areaDensity);

        numParticles = prim->verts.size();
    }



public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        auto areaDensity = get_input<zeno::NumericObject>("areaDensity")->get<int>();

        init(prim.get(), areaDensity);

        set_output("outPrim", std::move(prim));

    };
};

ZENDEFNODE(PBDClothInit, {// inputs:
                          {
                              {"PrimitiveObject", "prim"},
                              {"float", "areaDensity", "1.0"}
                          },
                          // outputs:
                          {
                              {"outPrim"}
                          },
                          // params:
                          {},
                          //category
                          {"PBD"}});

} // namespace zeno