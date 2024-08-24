#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/boundary_loop.h>
#include <igl/lscm.h>


namespace {
using namespace zeno;

struct CalPrimitiveUVMapLSCM : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        // int N = 4;
        
        size_t nm_vertices = prim->size();
        size_t nm_tris = prim->tris.size();

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        V.resize(nm_vertices,3);
        F.resize(nm_tris,3);

        const auto& verts = prim->verts;
        const auto& tris = prim->tris;

        for(int i = 0;i < nm_vertices;++i)
            V.row(i) << verts[i][0],verts[i][1],verts[i][2];
        
        for(int i = 0;i < nm_tris;++i)
            F.row(i) << tris[i][0],tris[i][1],tris[i][2];

        Eigen::VectorXi bnd,b(2,1);
        igl::boundary_loop(F,bnd);
        b(0) = bnd(0);
        b(1) = bnd(bnd.size()/2);

        Eigen::MatrixXd bc(2,2);
        bc<<0,0,1,0;


        Eigen::MatrixXd V_uv;
        igl::lscm(V,F,b,bc,V_uv);


        V_uv *= 5;

        auto& prim_uv = prim->add_attr<zeno::vec3f>("uv");
        for(int i = 0;i < V_uv.rows();++i)
            prim_uv[i] = zeno::vec3f(V_uv.row(i)[0],V_uv.row(i)[1],0);

        set_output("prim",prim);

    }

};

ZENDEFNODE(CalPrimitiveUVMapLSCM, {
    {gParamType_Primitive, "prim"},
    {gParamType_Primitive, "prim"},
    {},
    {"Parameterization"},
});

};