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

    float computeAngle( zeno::AttrVector<zeno::vec3f> &pos,
                        const zeno::AttrVector<zeno::vec4i> &tet,
                        int i)
    {
        auto id = vec4i(-1, -1, -1, -1);
        for (int j = 0; j < 4; j++)
            id[j] = tet[i][j];

        auto first = cross((pos[id[1]] - pos[id[0]]), (pos[id[2]] - pos[id[0]]));
        first = first / length(first) ;
        auto second = cross((pos[id[1]] - pos[id[0]]), (pos[id[3]] - pos[id[0]]));
        second = second / length(second) ;
        auto res = dot(first, second);
        return res;
    }

    void initRestLen(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &edge = prim->lines;
        for(int i = 0; i < edge.size(); i++)
            restLen[i] = length((pos[edge[i][0]] - pos[edge[i][1]]));
    }

    void initrestAng(PrimitiveObject *prim) 
    {
        auto &pos = prim->verts;
        auto &tris = prim->tris;
        auto &tet = prim->quads;
        for(int i = 0; i < tris.size(); i++)
            restAng[i] = computeAngle(pos, tet, i);
    }

    void initInvMass(PrimitiveObject *prim, float dx, float dy, float areaDensity) {
        auto &pos = prim->verts;
        auto &tet = prim->quads;
        for(int i = 0; i < tet.size(); i++)
        {
            float pInvMass = 0.0;
            pInvMass = areaDensity * dx * dy;
            for (int j = 0; j < 4; j++)
                invMass[tet[i][j]] += pInvMass;
        }
    }
    void initStiffMatrix(PrimitiveObject *prim)
    {
        //TODO:
    }

    void init(PrimitiveObject *prim, float dx, float dy, float areaDensity)
    {
        restLen.resize(prim->lines.size());
        invMass.resize(prim->verts.size());
        stiffMatrix.resize(prim->lines.size());

        initRestLen(prim);
        initrestAng(prim);
        initInvMass(prim, dx,  dy, areaDensity);

        numParticles = prim->verts.size();
    }



public:
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        //physical params
        auto externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();
        auto dt = get_input<zeno::NumericObject>("dt")->get<float>();
        auto edgeCompliance = get_input<zeno::NumericObject>("edgeCompliance")->get<float>();
        auto bendingCompliance = get_input<zeno::NumericObject>("bendingCompliance")->get<float>();

        //geometry params
        auto dx = get_input<zeno::NumericObject>("dx")->get<float>();
        auto dy = get_input<zeno::NumericObject>("dy")->get<float>();
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto areaDensity = get_input<zeno::NumericObject>("areaDensity")->get<int>();

        restLen = prim->lines->add_attr<float>("restLen");
        restAng = prim->tris->add_attr<float>("restAng");
        intMass = prim->verts->add_attr<float>("invMass");

        init(prim.get(), dx,  dy, areaDensity);

        set_output("outPrim", std::move(prim));

        set_output("dx", std::move(dx));
        set_output("dy", std::move(dy));
        set_output("nx", std::move(nx));
        set_output("ny", std::move(ny));

        set_output("externForce", std::move(externForce));
        set_output("dt", std::move(dt));
        set_output("edgeCompliance", std::move(edgeCompliance));
        set_output("bendingCompliance", std::move(bendingCompliance));

    };
};

ZENDEFNODE(PBDClothInit, {// inputs:
                          {
                              {"PrimitiveObject", "prim"},
                              {"vec3f", "externForce", "0.0, -10.0, 0.0"},
                              {"float", "dt", "0.001666667"},
                              {"float", "edgeCompliance", "100.0"},
                              {"float", "bendingCompliance", "1.0"},
                              {"float", "dx", "0.0078125"},
                              {"float", "dy", "0.0078125"},
                              {"int", "nx", "128"},
                              {"int", "ny", "128"},
                              {"float", "areaDensity", "1.0"},
                          },
                          // outputs:
                          {
                              {"outPrim"},
                              {"vec3f", "externForce"},
                              {"float", "dt", "0.001666667"},
                              {"float", "edgeCompliance"},
                              {"float", "bendingCompliance"},
                              {"float", "dx"},
                              {"float", "dy"},
                              {"int", "nx"},
                              {"int", "ny"},
                          },
                          // params:
                          {},
                          //category
                          {"PBD"}});

} // namespace zeno