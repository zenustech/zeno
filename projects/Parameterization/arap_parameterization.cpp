#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOFF.h>

#include <iostream>

namespace {
using namespace zeno;

struct CalPrimitiveUVMapARAP : zeno::INode {
    void set_boundary_condition(const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::VectorXi& bnd,
        Eigen::MatrixXd& bnd_uv) {

        }


    virtual void apply() override {
        #if 1
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

        #else
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            igl::readOFF("/home/lsl/Project/libigl/build/_deps/libigl_tutorial_tata-src/camelhead.off", V, F);
            auto prim = std::make_shared<zeno::PrimitiveObject>();
            prim->resize(V.rows());
            for(size_t i = 0;i < V.rows();++i)
                prim->verts[i] = zeno::vec3f(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
            prim->tris.resize(F.rows());
            for(size_t i = 0;i < F.rows();++i)
                prim->tris[i] = zeno::vec3i(F.row(i)[0],F.row(i)[1],F.row(i)[2]);

            // set_output("prim",prim);    
        #endif

        Eigen::VectorXi bnd;

        // std::cout << "boundary_loop" << std::endl;
        igl::boundary_loop(F,bnd);
        // std::cout << "boundary_condition" << std::endl;
        Eigen::MatrixXd bnd_uv;
        igl::map_vertices_to_circle(V,bnd,bnd_uv);
        // std::cout << "solve harmonic system for initial guess" << std::endl;
        Eigen::MatrixXd initial_guess;
        igl::harmonic(V,F,bnd,bnd_uv,1,initial_guess);

        igl::ARAPData arap_data;
        arap_data.with_dynamics = true;
        Eigen::VectorXi b = Eigen::VectorXi::Zero(0);
        Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0,0);

        std::cout << "arap precomputation" << std::endl;

        arap_data.max_iter = 100;
        igl::arap_precomputation(V,F,2,b,arap_data);

        std::cout << "solve arap " << std::endl;

        Eigen::MatrixXd V_uv = initial_guess;
        arap_solve(bc,arap_data,V_uv);

        V_uv *= 20;

        auto& prim_uv = prim->add_attr<zeno::vec3f>("uv");
        for(int i = 0;i < V_uv.rows();++i)
            prim_uv[i] = zeno::vec3f(V_uv.row(i)[0],V_uv.row(i)[1],0);

        set_output("prim",prim);

    }
};

ZENDEFNODE(CalPrimitiveUVMapARAP, {
    {"prim"},
    {"prim"},
    {},
    {"Parameterization"},
});

};