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
    virtual void apply() override {
        auto res = std::make_shared<PrimitiveObject>();
        auto mesh_path = get_input<zeno::StringObject>("mesh")->get();

        Eigen::MatrixXd V;
        Eigen::MatrixXi T,F;
        igl::readMESH(mesh_path,V,T,F);

        // we only support 3d simplex volumetric meshing
        assert(V.cols() == 3 && T.cols() == 4 && F.cols() == 3);

        res->resize(V.rows());
        for(size_t i = 0;i < V.rows();++i)
            res->verts[i] = zeno::vec3f(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
        for(size_t i = 0;i < T.rows();++i)
            res->quads.emplace_back(T.row(i)[0],T.row(i)[1],T.row(i)[2],T.row(i)[3]);

        for(size_t i = 0;i < res->quads.size();++i){
            auto tet = res->quads[i];
            res->tris.emplace_back(tet[0],tet[1],tet[2]);
            res->tris.emplace_back(tet[1],tet[3],tet[2]);
            res->tris.emplace_back(tet[0],tet[2],tet[3]);
            res->tris.emplace_back(tet[0],tet[3],tet[1]);
        }

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