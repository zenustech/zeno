#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>

namespace {
    using namespace zeno;

    struct CalPrimitveUVMapHarmonic : zeno::INode {
        virtual void apply() override {
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            auto order = get_param<int>("order");

            size_t nm_vertices = prim->size();
            size_t nm_tris= prim->tris.size();

            Eigen::MatrixXd V,V_uv;
            Eigen::MatrixXi F;
            V.resize(nm_vertices,3);
            F.resize(nm_tris,3);

            const auto& verts = prim->verts;
            const auto& tris = prim->tris;

            for(int i = 0;i < nm_vertices;++i)
                V.row(i) << verts[i][0],verts[i][1],verts[i][2];
            
            for(int i = 0;i < nm_tris;++i)
                F.row(i) << tris[i][0],tris[i][1],tris[i][2];

            Eigen::VectorXi bnd;
            igl::boundary_loop(F,bnd);

            // Map the boundary to a circle, preserving edge proportions
            Eigen::MatrixXd bnd_uv;
            igl::map_vertices_to_circle(V,bnd,bnd_uv);

            igl::harmonic(V,F,bnd,bnd_uv,order,V_uv);

            // Scale UV to make the texture more clear
            V_uv *= 5;                       

            auto& prim_uv = prim->add_attr<zeno::vec3f>("uv");
            for(int i = 0;i < V_uv.rows();++i)
                prim_uv[i] = zeno::vec3f(V_uv.row(i)[0],V_uv.row(i)[1],0);

            set_output("prim",prim);            

        }
    };

    ZENDEFNODE(CalPrimitveUVMapHarmonic, {
        {gParamType_Primitive, "prim"},
        {gParamType_Primitive, "prim"},
        {{gParamType_Int,"order","1"}},
        {"Parameterization"},
    });

}