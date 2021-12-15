#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>


#include <igl/lbs_matrix.h>

#include "skinning_iobject.h"

namespace{
using namespace zeno;

struct PosesAnimationFrame;

struct MakeRestPoses : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<PosesAnimationFrame>();
        auto bones = get_input<zeno::PrimitiveObject>("bones");

        res->posesFrame.resize(bones->lines.size(),Eigen::Quaterniond::Identity());

        set_output("posesFrame",std::move(res));
    }
};

ZENDEFNODE(MakeRestPoses, {
    {"bones"},
    {"posesFrame"},
    {},
    {"Skinning"},
});

struct BlendPoses : zeno::INode {
    virtual void apply() override {
        auto poses1 = get_input<PosesAnimationFrame>("p1");
        auto poses2 = get_input<PosesAnimationFrame>("p2");

        // if(poses1->posesFrame.cols() != 4 || poses1->posesFrame.cols() != 4)
        //     throw std::runtime_error("INVALIED POSES DIMENSION");

        if(poses1->posesFrame.size() != poses2->posesFrame.size()){
            std::cout << "THE DIMENSION OF TWO MERGED POSES DOES NOT MATCH " << poses1->posesFrame.size() << "\t" << poses2->posesFrame.size() << std::endl;
            throw std::runtime_error("THE DIMENSION OF TWO MERGED POSES DOES NOT MATCH");
        }

        auto w = get_input<zeno::NumericObject>("w")->get<float>();

        auto res = std::make_shared<PosesAnimationFrame>();
        res->posesFrame.resize(poses1->posesFrame.size());

        for(size_t i = 0;i < res->posesFrame.size();++i){
            res->posesFrame[i] =  poses1->posesFrame[i].slerp(1-w,poses2->posesFrame[i]);
        }

        std::cout << "OUT_POSES : " << std::endl;
        for(size_t i = 0;i < res->posesFrame.size();++i)
            std::cout << "P<" << i << "> : " << res->posesFrame[i] << std::endl;

        set_output("bp",std::move(res));
    }
};

ZENDEFNODE(BlendPoses, {
    {"p1","p2","w"},
    {"bp"},
    {},
    {"Skinning"},
});


struct DoSkinning : zeno::INode {
    virtual void apply() override {
        std::cout << "DO LINEAR BLEND SKINNING" << std::endl;

        auto shape = get_input<PrimitiveObject>("shape");
        auto pose = get_input<PosesAnimationFrame>("pose");
        auto bones = get_input<PrimitiveObject>("bones");
        auto W = get_input<SkinningWeight>("W");

        auto algorithm = std::get<std::string>(get_param("algorithm"));

        RotationList vQ;
        std::vector<Eigen::Vector3d> vT;

        Eigen::MatrixXd C;
        Eigen::MatrixXi BE;

        C.resize(bones->size(),3);
        BE.resize(bones->lines.size(),2);

        for(size_t i = 0;i < bones->size();++i){
            C.row(i) << bones->verts[i][0],bones->verts[i][1],bones->verts[i][2];
        }

        for(size_t i = 0;i < bones->lines.size();++i)
            BE.row(i) << bones->lines[i][0],bones->lines[i][1];


        Eigen::VectorXi P;
        igl::directed_edge_parents(BE,P);

        // std::cout << "DO FORWARD KINEMATICS" << std::endl;
        igl::forward_kinematics(C,BE,P,pose->posesFrame,vQ,vT);

        const int dim = C.cols();
        Eigen::MatrixXd T(BE.rows()*(dim+1),dim);
        for(int e = 0;e<BE.rows();e++){
            Eigen::Affine3d a = Eigen::Affine3d::Identity();
            a.translate(vT[e]);
            a.rotate(vQ[e]);
            T.block(e*(dim+1),0,dim+1,dim) =
                a.matrix().transpose().block(0,0,dim+1,dim);
        }
        std::cout << "COMPUTE DEFORMATION VIA LBS" << std::endl;
        // Compute deformation via LBS as matrix multiplication
        Eigen::MatrixXd U,V;
        V.resize(shape->size(),3);
        for(size_t i = 0;i < V.rows();++i)
            V.row(i) << shape->verts[i][0],shape->verts[i][1],shape->verts[i][2];

        if(algorithm == "DQS"){
            igl::dqs(V,W->weight,vQ,vT,U);
        }else if(algorithm == "LBS"){
            Eigen::MatrixXd M;
            igl::lbs_matrix(V,W->weight,M);
            U = M*T;
        }        
        // std::cout << "U : " << U.rows() << "\t" << U.cols() << std::endl;
        // std::cout << "BLSW : " << blsw->weight.rows() << "\t" << blsw->weight.cols() << std::endl;
        // std::cout << "T : " << T.rows() << "\t" << T.cols() << std::endl;

        auto deformed_shape = std::make_shared<zeno::PrimitiveObject>();
        deformed_shape->resize(shape->size());
        deformed_shape->tris.resize(shape->tris.size());
        deformed_shape->quads.resize(shape->quads.size());

        for(size_t i = 0;i < deformed_shape->size();++i)
            deformed_shape->verts[i] = zeno::vec3f(U.row(i)[0],U.row(i)[1],U.row(i)[2]);

        for(size_t i = 0;i < shape->tris.size();++i)
            deformed_shape->tris[i] = shape->tris[i];

        for(size_t i = 0;i < shape->quads.size();++i)
            deformed_shape->quads[i] = shape->quads[i];

        // Also deform skeleton edges
        Eigen::MatrixXd CT;
        Eigen::MatrixXi BET;
        igl::deform_skeleton(C,BE,T,CT,BET);

        auto deformed_bones = std::make_shared<zeno::PrimitiveObject>();
        deformed_bones->resize(CT.rows());
        deformed_bones->lines.resize(BET.rows());

        for(size_t i = 0;i < CT.rows();++i)
            deformed_bones->verts[i] = zeno::vec3f(CT.row(i)[0],CT.row(i)[1],CT.row(i)[2]);
        for(size_t i = 0;i < BET.rows();++i)
            deformed_bones->lines[i] = zeno::vec2i(BET.row(i)[0],BET.row(i)[1]);

        set_output("dshape",std::move(deformed_shape));
        set_output("dbones",std::move(deformed_bones));

        std::cout << "FINISH OUTPUT DEFORMED SHAPE AND BONES" << std::endl;
    }
};

ZENDEFNODE(DoSkinning, {
    {"shape","pose","bones","W"},
    {"dshape","dbones"},
    {{"enum LBS DQS","algorithm","LBS"}},
    {"Skinning"},
});

};