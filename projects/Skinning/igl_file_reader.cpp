#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/readOBJ.h>
#include <igl/readTGF.h>
#include <igl/readMESH.h>

#include "skinning_iobject.h"

namespace{
using namespace zeno;

struct LoadTFGPrimitiveFromFile : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<zeno::PrimitiveObject>();
        auto tfg_path = get_input<zeno::StringObject>("tfg")->get();

        Eigen::MatrixXd C;
        Eigen::MatrixXi BE;    
        igl::readTGF(tfg_path,C,BE);

        assert(C.cols() == 3);
        assert(BE.cols() == 2);


        // auto& parents = res->add_attr<int>("parents");
        res->resize(C.rows());
        auto& pos = res->attr<zeno::vec3f>("pos");
        auto& segs = res->lines;
        segs.resize(BE.rows());

        for(size_t i = 0;i < C.rows();++i)
            pos[i] = zeno::vec3f(C.row(i)[0],C.row(i)[1],C.row(i)[2]);
        for(size_t i = 0;i < BE.rows();++i)
            segs[i] = zeno::vec2i(BE.row(i)[0],BE.row(i)[1]);

        // Eigen::VectorXi P;
        // igl::directed_edge_parents(BE,P);

        // std::cout << "BE : " << std::endl << BE << std::endl;
        // std::cout << "P : " << std::endl << P.transpose() << std::endl;

        // std::cout << "P : " << P.rows() << "\t" << P.cols() << std::endl;
        // std::cout << "parents : " << parents.size() << std::endl;

        // for(size_t i = 0;i < parents.size();++i)
        //     parents[i] = P[i];

        set_output("res",std::move(res));
    }
}; 

ZENDEFNODE(LoadTFGPrimitiveFromFile, {
    {{"readpath","tfg"}},
    {"res"},
    {},
    {"Skinning"},
});



struct ReadMesh : zeno::INode {
    static constexpr auto s_pi = 3.1415926535897932384626433832795028841972L;
    static constexpr auto s_half_pi = 1.5707963267948966192313216916397514420986L;
    virtual void apply() override {
        auto mesh_path = get_input<zeno::StringObject>("mesh")->get();
        Eigen::MatrixXd V;
        Eigen::MatrixXi T,F;
        igl::readMESH(mesh_path,V,T,F);

        // we only support 3d simplex volumetric meshing
        assert(V.cols() == 3 && T.cols() == 4 && F.cols() == 3);

        auto res = std::make_shared<PrimitiveObject>();
        auto &stag = res->add_attr<float>("stag");
        auto &fiberDir = res->add_attr<zeno::vec3f>("fiberDir");
        res->resize(V.rows());

        for(size_t i = 0;i < V.rows();++i)
            res->verts[i] = zeno::vec3f(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
        for(size_t i = 0;i < T.rows();++i)
            res->quads.emplace_back(T.row(i)[0],T.row(i)[1],T.row(i)[2],T.row(i)[3]);

        std::vector<float> srs;
        srs.resize(V.rows(),0);

        for(size_t i = 0;i < res->quads.size();++i){
            auto tet = res->quads[i];
            res->tris.emplace_back(tet[0],tet[1],tet[2]);
            res->tris.emplace_back(tet[1],tet[3],tet[2]);
            res->tris.emplace_back(tet[0],tet[2],tet[3]);
            res->tris.emplace_back(tet[0],tet[3],tet[1]);

            for(size_t k = 0;k < 4;++k){
                size_t l = (k+1) % 4;
                size_t m = (k+2) % 4;
                size_t n = (k+3) % 4;

                auto v0 = res->verts[tet[k]];
                auto v1 = res->verts[tet[l]];
                auto v2 = res->verts[tet[m]];
                auto v3 = res->verts[tet[n]];

                auto v10 = v1 - v0;
                auto v20 = v2 - v0;
                auto v30 = v3 - v0;

                auto l10 = zeno::length(v10);
                auto l20 = zeno::length(v20);
                auto l30 = zeno::length(v30);

                auto alpha = zeno::acos(zeno::dot(v10,v20)/l10/l20);
                auto beta = zeno::acos(zeno::dot(v10,v30)/l10/l30);
                auto gamma = zeno::acos(zeno::dot(v20,v30)/l20/l30);

                auto s = 0.5 * (alpha + beta + gamma);

                auto omega = 4*zeno::atan(zeno::sqrt(zeno::tan(s/2)*zeno::tan((s - alpha)/2)*zeno::tan((s-beta)/2)*zeno::tan((s-gamma)/2)));

                srs[tet[k]] += omega;
            }
        }
        // for interior points, the surrounded solid angles should sum up to 4*pi
        std::fill(stag.begin(),stag.end(),0.0);
        size_t nm_surface_verts = 0;
        for(size_t i = 0;i < V.rows();++i){
            // std::cout << "STAG<" << i << ">\t: " << srs[i] << "\t" << 4*s_pi << std::endl;
            if(zeno::abs(srs[i] - 4*s_pi) > 1e-3){
                stag[i] = 1.0;
                nm_surface_verts++;
            }

            fiberDir[i] = zeno::vec3f(0.0,1.0,0.0);
        }
        // std::cout << "NM_SURFACE_VERTS : " << nm_surface_verts << std::endl;

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(ReadMesh, {
    {{"readpath","mesh"}},
    {"res"},
    {},
    {"Skinning"},
});

};