#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/boundary_conditions.h>
#include <igl/bbw.h>
#include <igl/normalize_row_sums.h>

#include "skinning_iobject.h"

namespace{
using namespace zeno;




struct GenerateSkinningWeight : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("skinMesh");
        auto tfg = get_input<PrimitiveObject>("skinBone");

        Eigen::MatrixXd V;
        Eigen::MatrixXi T;
        Eigen::MatrixXd C;
        Eigen::MatrixXi BE;

        // first we need to transfer the data from zeno's data structure to igl's
        // we only support 3d space only and tetradedron volume mesh
        V.resize(mesh->size(),3);
        T.resize(mesh->quads.size(),4);

        for(size_t i = 0;i < mesh->size();++i)
            V.row(i) << mesh->verts[i][0],mesh->verts[i][1],mesh->verts[i][2];
        for(size_t i = 0;i < mesh->quads.size();++i)
            T.row(i) << mesh->quads[i][0],mesh->quads[i][1],mesh->quads[i][2],mesh->quads[i][3];

        C.resize(tfg->size(),3);
        BE.resize(tfg->lines.size(),3);

        // do some vertices alignment here
        for(size_t i = 0;i < tfg->size();++i){
            Eigen::Vector3d tfg_vert;tfg_vert << tfg->verts[i][0],tfg->verts[i][1],tfg->verts[i][2];
            Eigen::Vector3d align_vert = tfg_vert;

            double min_dist = 1e18;
            for(size_t j = 0;j < mesh->size();++j){
                Eigen::Vector3d mvert;mvert << mesh->verts[j][0],mesh->verts[j][1],mesh->verts[j][2];
                double dist = (mvert - tfg_vert).norm();
                if(dist < min_dist){
                    min_dist = dist;
                    align_vert = mvert;
                }
            }
            C.row(i) = align_vert;
        }
        for(size_t i = 0;i < tfg->lines.size();++i)
            BE.row(i) << tfg->lines[i][0],tfg->lines[i][1];

        Eigen::VectorXi b;
        // List of boundary conditions of each weight function
        Eigen::MatrixXd bc;
        igl::boundary_conditions(V,T,C,Eigen::VectorXi(),BE,Eigen::MatrixXi(),b,bc);


        // std::cout << "OUTPUT B AND BC : " << std::endl;
        // std::cout << "B : " << std::endl << b << std::endl;
        // std::cout << "BC : " << std::endl << bc << std::endl;

          // compute BBW weights matrix
        igl::BBWData bbw_data;
        // only a few iterations for sake of demo
        bbw_data.active_set_params.max_iter = 8;
        bbw_data.verbosity = 1;

        Eigen::MatrixXd W;
        if(!igl::bbw(V,T,b,bc,bbw_data,W))
        {
            throw std::runtime_error("BBW GENERATION FAIL");
        }

        assert(W.rows() == V.rows() && W.cols() == C.rows());

        auto res = std::make_shared<SkinningWeight>();

        igl::normalize_row_sums(W,W);

        res->weight = W;


        std::cout << "OUTPUT_W" << std::endl;

        // add weight channel
        for(size_t i = 0;i < W.cols();++i){
            std::string channel_name = "sw_" + std::to_string(i);
            auto& c = mesh->add_attr<float>(channel_name);
            mesh->resize(V.rows());

            for(size_t j = 0;j < W.rows();++j){
                c[j] = W(j,i);
                // std::cout << W(j,0) << std::endl;
            }

            // for(size_t i = 0;i < V.rows();++i)
            //     std::cout << "<" << i << "> : " << mesh->attr<float>(channel_name)[i] << std::endl;
        }

        set_output("W",std::move(res));
        set_output("mesh",mesh);
    }
};

ZENDEFNODE(GenerateSkinningWeight, {
    {"skinMesh","skinBone"},
    {"W","mesh"},
    {},
    {"Skinning"},
});

};