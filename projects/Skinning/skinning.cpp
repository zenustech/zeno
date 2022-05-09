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
    template <
    typename DerivedV,
    typename DerivedW,
    typename DerivedP,
    typename Q,
    typename QAlloc,
    typename T,
    typename DerivedU>    
    void CoRs(
        const Eigen::MatrixBase<DerivedV> & V,
        const Eigen::MatrixBase<DerivedW> & W,
        const Eigen::MatrixBase<DerivedP> & P,
        const std::vector<Q,QAlloc> & vQ,
        const std::vector<T> & vT,
        Eigen::PlainObjectBase<DerivedU> & U)
    {
        using namespace std;
        assert(V.rows() <= W.rows());
        assert(W.cols() == (int)vQ.size());
        assert(W.cols() == (int)vT.size());
        // resize output
        U.resizeLike(V);

        const int nv = V.rows();
    #pragma omp parallel for if (nv > 10000)
        for(size_t i = 0;i < nv;++i){

        }
    }

    virtual void apply() override {
        auto shape = get_input<PrimitiveObject>("shape");
        auto algorithm = std::get<std::string>(get_param("algorithm"));
        auto attr_prefix = get_param<std::string>("attr_prefix");
        auto outputChannel = get_param<std::string>("out_channel");

        auto Qs_ = get_input<zeno::ListObject>("Qs")->get<std::shared_ptr<NumericObject>>();
        auto Ts_ = get_input<zeno::ListObject>("Ts")->get<std::shared_ptr<NumericObject>>();

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
        }else if(algorithm == "CoRs"){
            Eigen::MatrixXd M;
            Eigen::MatrixXd R = Eigen::MatrixXd(V.rows(),V.cols());
            int nv = V.rows();
            const auto& CoRs = shape->attr<zeno::vec3f>("rCenter");
        #pragma omp parallel for if (nv > 10000)
            for(int i = 0;i < nv;++i)
                R.row(i) << CoRs[i][0],CoRs[i][1],CoRs[i][2];
            igl::lbs_matrix(R,W,M);
            Eigen::MatrixXd TP = M*T;

        #pragma omp parallel for if (nv > 10000)
            for(int i = 0;i < nv;++i){
                // compute the blending quaternion
                Eigen::Quaterniond b(0,0,0,0);
                for(int c = 0;c < W.cols();++c)
                    b.coeffs() += W(i,c) * Qs[c].coeffs();
                Eigen::Quaterniond c = b;
                c.coeffs() /= b.norm();

                Eigen::Vector3d v = R.row(i);
                Eigen::Vector3d d = c.vec();
                double a = c.w();
                TP.row(i) -= (v + 2 * d.cross(d.cross(v) + a*v));

                // transform the quaternion and translation into dual parts
                Eigen::Quaterniond q = c;
                Eigen::Quaterniond dq = Eigen::Quaterniond(0,0,0,0);
                const auto& t = TP.row(i);
                dq.w() = -0.5*( t(0)*q.x() + t(1)*q.y() + t(2)*q.z());
                dq.x() =  0.5*( t(0)*q.w() + t(1)*q.z() - t(2)*q.y());
                dq.y() =  0.5*(-t(0)*q.z() + t(1)*q.w() + t(2)*q.x());
                dq.z() =  0.5*( t(0)*q.y() - t(1)*q.x() + t(2)*q.w());

                q.coeffs() /= q.norm();
                dq.coeffs() /= dq.norm();

                v = V.row(i);
                Eigen::Vector3d d0 = q.vec();
                Eigen::Vector3d de = dq.vec();
                double a0 = q.w();
                double ae = dq.w();
                U.row(i) =  v + 2*d0.cross(d0.cross(v) + a0*v) + 2*(a0*de - ae*d0 + d0.cross(de));
            } 
        }else{
            std::cerr << "INVALID ALGOROTHM SPECIFIED " << algorithm << std::endl;
            throw std::runtime_error("DoSkinning : INVALID ALGORITHM SPECIFIED");
        }        

        // auto deformed_shape = std::make_shared<zeno::PrimitiveObject>(*shape);// automatic copy all the attributes
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
        auto& out_chan = shape->add_attr<zeno::vec3f>(outputChannel);
        for(size_t i = 0;i < shape->size();++i)
            out_chan[i] = zeno::vec3f(U.row(i)[0],U.row(i)[1],U.row(i)[2]);

        // std::cout << "U:" << U.row(2) << "\t" << deformed_shape->size() << std::endl;

        set_output("shape",shape);
    }
};

ZENDEFNODE(DoSkinning, {
    {"shape","Qs","Ts","restBones"},
    {"shape"},
    {{"enum LBS DQS","algorithm","DQS"},{"string","attr_prefix","sw"},{"string","out_channel","curPos"},{"int","FK","0"}},
    {"Skinning"},
});

};