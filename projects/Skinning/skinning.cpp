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
        set_output("bp",std::move(res));
    }
};

ZENDEFNODE(BlendPoses, {
    {"p1","p2","w"},
    {"bp"},
    {},
    {"Skinning"},
});

// input the forward kinematics result
struct DoSkinning : zeno::INode {
    virtual void apply() override {
        auto shape = get_input<PrimitiveObject>("shape");
        auto algorithm = get_param<std::string>(("algorithm"));
        auto attr_prefix = get_param<std::string>("attr_prefix");

        auto Qs_ = get_input<zeno::ListObject>("Qs")->get<NumericObject>();
        auto Ts_ = get_input<zeno::ListObject>("Ts")->get<NumericObject>();

        // std::cout << "GOT QS AND TS INPUT" << std::endl;
        size_t dim = 3;
        size_t nm_handles = 0;

        // std::cout << "CHECKOUT_1" << std::endl;

        while(true){
            std::string attr_name = attr_prefix + "_" + std::to_string(nm_handles);
            if(shape->has_attr(attr_name)){
                nm_handles++;
                continue;
            }
            break;
        }

        Eigen::MatrixXd W;
        W.resize(shape->size(),nm_handles);
        for(size_t i = 0;i < nm_handles;++i){ 
            std::string attr_name = attr_prefix +  "_" + std::to_string(i);
            for(size_t j = 0;j < shape->size();++j){
                if(!shape->has_attr(attr_name)){
                    std::cout << "DO NOT HAVE " << attr_name << std::endl;
                    std::cout << "NM_QS_AND_TS : " << nm_handles << std::endl;
                    throw std::runtime_error("The Skinned Prim Does Not Have Weight Attr");
                }
                W(j,i) = shape->attr<float>(attr_name)[j];
                if(std::isnan(W(j,i))){
                    std::cout << "NAN VALUE DETECTED IN SKINNING WEIGHT MATRIX : " << j << "\t" << i << "\t" << W(j,i) << std::endl;
                    throw std::runtime_error("NAN VALUE DETECTED IN SKINNING WEIGHT MATRIX");
                }
            }
        }

        std::vector<Eigen::Vector3d> Ts;
        RotationList Qs;

        auto do_FK = get_param<int>("FK");
        if(!do_FK){
            std::cout << "GLOBAL TRANSFORMATION BLENDING" << std::endl;
            std::cout << "NM_HANDLES : " << nm_handles << std::endl;
            for(size_t i = 0;i < nm_handles;++i){
                if(std::isnan(zeno::length(Ts_[i]->get<zeno::vec3f>())) || std::isnan(zeno::length(Qs_[i]->get<zeno::vec4f>()))){
                    std::cout << "NAN RIGGING AFFINE TRANSFORMATION DETECTED" << std::endl;
                    std::cout << "T<" << i << "> : " << Eigen::Vector3d(Ts_[i]->get<zeno::vec3f>()[0],
                        Ts_[i]->get<zeno::vec3f>()[1],
                        Ts_[i]->get<zeno::vec3f>()[2]).transpose() << std::endl;

                    std::cout << "Q<" << i << "> : " << Eigen::Vector4d(Qs_[i]->get<zeno::vec4f>()[0],
                        Qs_[i]->get<zeno::vec4f>()[1],
                        Qs_[i]->get<zeno::vec4f>()[2],
                        Qs_[i]->get<zeno::vec4f>()[3]).transpose() << std::endl;

                    throw std::runtime_error("NAN RIGGING AFFINE TRANSFORMATION DETECTED");
                }

                Ts.emplace_back(Ts_[i]->get<zeno::vec3f>()[0],
                    Ts_[i]->get<zeno::vec3f>()[1],
                    Ts_[i]->get<zeno::vec3f>()[2]);
                Qs.emplace_back(Qs_[i]->get<zeno::vec4f>()[3],
                    Qs_[i]->get<zeno::vec4f>()[0],
                    Qs_[i]->get<zeno::vec4f>()[1],
                    Qs_[i]->get<zeno::vec4f>()[2]);
            }
        }else{
            if(!has_input("restBones")){
                throw std::runtime_error("INPUT JOINTS INFOR FOR FORWARD KINEMATICS");
            }
            auto bones = get_input<PrimitiveObject>("restBones");
            std::vector<Eigen::Vector3d> LFT;
            RotationList LFQ;
            for(size_t i = 0;i < nm_handles;++i){
                LFT.emplace_back(Ts_[i]->get<zeno::vec3f>()[0],
                    Ts_[i]->get<zeno::vec3f>()[1],
                    Ts_[i]->get<zeno::vec3f>()[2]);
                LFQ.emplace_back(Qs_[i]->get<zeno::vec4f>()[3],
                    Qs_[i]->get<zeno::vec4f>()[0],
                    Qs_[i]->get<zeno::vec4f>()[1],
                    Qs_[i]->get<zeno::vec4f>()[2]);
            }

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
            igl::forward_kinematics(C,BE,P,LFQ,LFT,Qs,Ts);
        }



        // std::cout << "CHECKOUT_3" << std::endl;

        Eigen::MatrixXd T(nm_handles*(dim+1),dim);
        for(int e = 0;e<nm_handles;e++){
            Eigen::Affine3d a = Eigen::Affine3d::Identity();
            a.translate(Ts[e]);
            a.rotate(Qs[e]);
            T.block(e*(dim+1),0,dim+1,dim) =
                a.matrix().transpose().block(0,0,dim+1,dim);
        }
        // Compute deformation via LBS as matrix multiplication
        Eigen::MatrixXd U,V;
        V.resize(shape->size(),3);
        for(size_t i = 0;i < V.rows();++i)
            V.row(i) << shape->verts[i][0],shape->verts[i][1],shape->verts[i][2];


        if(std::isnan(V.norm()) || std::isnan(W.norm()) || std::isnan(T.norm())){
            std::cout << V.norm() << "\t" << W.norm() << std::endl;
            throw std::runtime_error("IN SKINNING NAN VW DETECTED");
        }

        if(algorithm == "DQS"){
            // std::cout << "DQS SKINNING " << std::endl;
            igl::dqs(V,W,Qs,Ts,U);
        }else if(algorithm == "LBS"){
            Eigen::MatrixXd M;
            igl::lbs_matrix(V,W,M);
            U = M*T;
        }        

        auto deformed_shape = std::make_shared<zeno::PrimitiveObject>(*shape);// automatic copy all the attributes
        // deformed_shape->resize(shape->size());
        // deformed_shape->tris.resize(shape->tris.size());
        // deformed_shape->quads.resize(shape->quads.size());

        if(std::isnan(U.norm())){
            std::cout << "W : \n" << W << std::endl;
            std::cout << "NAN DEFORMED SHAPE DETECTED: " << U.norm() << std::endl;
            std::cout << "AFFINE : " << std::endl;
            for(size_t i = 0;i < nm_handles;++i){
                std::cout << Qs[i].x() << "\t" 
                            << Qs[i].y() << "\t" 
                            << Qs[i].z() << "\t" 
                            << Qs[i].w() << std::endl;
                std::cout << Ts[i].transpose() << std::endl;
            }

            throw std::runtime_error("NAN DEFORMED SHAPE DETECTED");
        }


        // std::cout << "CHECKOUT_4" << std::endl;

        for(size_t i = 0;i < deformed_shape->size();++i)
            deformed_shape->verts[i] = zeno::vec3f(U.row(i)[0],U.row(i)[1],U.row(i)[2]);

        set_output("dshape",std::move(deformed_shape));
    }
};

ZENDEFNODE(DoSkinning, {
    {"shape","Qs","Ts","restBones"},
    {"dshape"},
    {{"enum LBS DQS","algorithm","DQS"},{"string","attr_prefix","sw"},{"int","FK","0"}},
    {"Skinning"},
});

};
