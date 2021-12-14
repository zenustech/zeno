#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/readOBJ.h>

#include <igl/readTGF.h>
#include <igl/boundary_conditions.h>
#include <igl/bbw.h>
#include <igl/lbs_matrix.h>
#include <igl/readMESH.h>

#include <igl/normalize_row_sums.h>
#include <igl/readDMAT.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>

#include <igl/forward_kinematics.h>
#include <igl/deform_skeleton.h>

#include <igl/dqs.h>

#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Transform.h>


namespace{
using namespace zeno;


typedef
  std::vector<Eigen::Quaterniond,Eigen::aligned_allocator<Eigen::Quaterniond> >
  RotationList;

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


struct RetrieveTFGByFromPrim : zeno::INode {}

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


struct SkinningWeight : zeno::IObject {
    SkinningWeight() = default;
    Eigen::MatrixXd weight;
};

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

        for(size_t i = 0;i < tfg->size();++i)
            C.row(i) << tfg->verts[i][0],tfg->verts[i][1],tfg->verts[i][2];
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

struct BBW2LBSW : zeno::INode {
    virtual void apply() override {
        auto shape = get_input<zeno::PrimitiveObject>("skinMesh");
        auto bbw = get_input<SkinningWeight>("bbw");
        auto lbsw = std::make_shared<SkinningWeight>();

        Eigen::MatrixXd V;
        V.resize(shape->size(),3);
        for(size_t i = 0;i < V.rows();++i)
            V.row(i) << shape->verts[i][0],shape->verts[i][1],shape->verts[i][2];
        // compute linear blend skinning matrix
        igl::lbs_matrix(V,bbw->weight,lbsw->weight);

        set_output("lbsw",std::move(lbsw));
    }
};

ZENDEFNODE(BBW2LBSW, {
    {"skinMesh","bbw"},
    {"lbsw"},
    {},
    {"Skinning"},
});


struct PosesAnimationFrame : zeno::IObject {
    PosesAnimationFrame() = default;
    RotationList posesFrame;
};

struct ReadPoseFrame : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<PosesAnimationFrame>();
        auto bones = get_input<zeno::PrimitiveObject>("bones");
        auto dmat_path = get_input<zeno::StringObject>("dmat_path")->get();

        Eigen::MatrixXd Q;
        igl::readDMAT(dmat_path,Q);

        if(bones->lines.size() != Q.rows()/4 || Q.rows() % 4 != 0){
            std::cout << "THE DIMENSION OF BONES DOES NOT MATCH POSES  " << bones->lines.size() << "\t" << Q.rows() << std::endl;
            throw std::runtime_error("THE DIMENSION OF BONES DOES NOT MATCH POSES");
        }

        igl::column_to_quats(Q,res->posesFrame);
        set_output("posesFrame",std::move(res));
    }
};

ZENDEFNODE(ReadPoseFrame, {
    {{"readpath","dmat_path"},"bones"},
    {"posesFrame"},
    {},
    {"Skinning"},
});

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
