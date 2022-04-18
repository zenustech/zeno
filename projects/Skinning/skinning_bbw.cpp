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
#include <igl/project_to_line.h>

#include "skinning_iobject.h"


namespace{
using namespace zeno;

struct SolveBiharmonicWeight : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("skinMesh");
        auto nm_handles = (int)get_input2<float>("nm_handles");
        if(!mesh->has_attr("BoneID")){
            throw std::runtime_error("SkinMesh should have BoneID indicating the binding vertices' to the handles");
        }
        // btag == -1 means free vertices, and (int)indicating the index of binding handle
        const auto& boneIDs = mesh->attr<float>("BoneID");
        auto attr_prefix = get_param<std::string>("attr_prefix");

        Eigen::MatrixXd V(mesh->size(),3);
        Eigen::MatrixXi T(mesh->quads.size(),4);


        size_t nm_boundary_verts = 0;
        for(size_t i = 0;i < mesh->size();++i){
            V.row(i) << mesh->verts[i][0],mesh->verts[i][1],mesh->verts[i][2];
            if(boneIDs[i] > -1e-6)
                nm_boundary_verts++;
        }
        for(size_t i = 0;i < mesh->quads.size();++i){
            T.row(i) << mesh->quads[i][0],mesh->quads[i][1],mesh->quads[i][2],mesh->quads[i][3];
        }

        Eigen::VectorXi b(nm_boundary_verts);
        // List of boundary conditions of each weight function
        Eigen::MatrixXd bc(nm_boundary_verts,nm_handles);bc.setZero();

        size_t b_idx = 0;
        for(size_t i = 0;i < mesh->size();++i){
            if(boneIDs[i] > -1e-6){
                int handle_idx = (int)boneIDs[i];
                b[b_idx] = i;
                bc(b_idx,handle_idx) = 1.0;
                ++b_idx;
            }
        }
        std::cout << "BBW: size of bc " << bc.rows() << "\t" << bc.cols() << std::endl;
        // compute BBW weights matrix
        igl::BBWData bbw_data;
        // only a few iterations for sake of demo
        bbw_data.active_set_params.max_iter = 8;
        bbw_data.verbosity = 0;

        Eigen::MatrixXd W;
        if(!igl::bbw(V,T,b,bc,bbw_data,W))
        {
            throw std::runtime_error("BBW GENERATION FAIL");
        }
        assert(W.rows() == V.rows() && W.cols() == C.rows());
        igl::normalize_row_sums(W,W);

        for(size_t i = 0;i < W.cols();++i){
            std::string channel_name = attr_prefix + "_" + std::to_string(i);
            auto& c = mesh->add_attr<float>(channel_name);
            for(size_t j = 0;j < W.rows();++j){
                c[j] = W(j,i);
            }
        }
        set_output("mesh",mesh);
    }
};

ZENDEFNODE(SolveBiharmonicWeight, {
    {"skinMesh",{"float","nm_handles","2"}},
    {"mesh"},
    {
        {"string","attr_prefix","sw"},
    },
    {"Skinning"},
});


struct GenerateSkinningWeight : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<PrimitiveObject>("skinMesh");
        auto tfg = get_input<PrimitiveObject>("skinBone");
        auto attr_prefix = get_param<std::string>("attr_prefix");
        // auto BonesID = get_input2<int>("BID");
        auto sp_influence_radius = get_param<float>("sp_radius");
        auto bone_influence_radius = get_param<float>("bone_radius");
        auto cage_influence_radius = get_param<float>("cage_radius");
        // auto duplicate = (int)get_param<float>("duplicate");
        // auto bone_type = get_param<std::string>("boneType"));
        auto duplicate = (int)get_param<float>("duplicate");
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
        BE.resize(tfg->lines.size(),2);

        // do some vertices alignment here
        for(size_t i = 0;i < tfg->size();++i){
            Eigen::Vector3d tfg_vert;tfg_vert << tfg->verts[i][0],tfg->verts[i][1],tfg->verts[i][2];
            // Eigen::Vector3d align_vert = tfg_vert;
            // double min_dist = 1e18;
            // for(size_t j = 0;j < mesh->size();++j){
            //     Eigen::Vector3d mvert;mvert << mesh->verts[j][0],mesh->verts[j][1],mesh->verts[j][2];
            //     double dist = (mvert - tfg_vert).norm();
            //     if(dist < min_dist){
            //         min_dist = dist;
            //         align_vert = mvert;
            //     }
            // }
            // C.row(i) = align_vert;
            C.row(i) << tfg->verts[i][0],tfg->verts[i][1],tfg->verts[i][2];
        }
        for(size_t i = 0;i < tfg->lines.size();++i)
            BE.row(i) << tfg->lines[i][0],tfg->lines[i][1];

        Eigen::VectorXi b;
        // List of boundary conditions of each weight function
        Eigen::MatrixXd bc;
        // TODO : rewrite this part of function so that a radius parameter could be included
        // igl::boundary_conditions(V,T,C,Eigen::VectorXi(),BE,Eigen::MatrixXi(),b,bc);

        // std::cout << "BE : " << std::endl << BE << std::endl;
        boundary_conditions(V,T,C,Eigen::VectorXi(),BE,Eigen::MatrixXi(),b,bc,sp_influence_radius,bone_influence_radius,cage_influence_radius);

        // std::cout << "nm BINDING POINTS : " << b.size() << std::endl;

        // compute BBW weights matrix
        igl::BBWData bbw_data;
        // only a few iterations for sake of demo
        bbw_data.active_set_params.max_iter = 8;
        bbw_data.verbosity = 0;

        std::cout << "SIZE OF BC : " << bc.rows() << "\t" << bc.cols() << std::endl;

        Eigen::MatrixXd W;
        if(!igl::bbw(V,T,b,bc,bbw_data,W))
        {
            throw std::runtime_error("BBW GENERATION FAIL");
        }

        assert(W.rows() == V.rows() && W.cols() == C.rows());
        igl::normalize_row_sums(W,W);

        // add weight channel

        for(size_t i = 0;i < W.cols()/duplicate;++i){
            std::string channel_name = attr_prefix + "_" + std::to_string(i);
            std::cout << "add channels " << channel_name << std::endl;
            auto& c = mesh->add_attr<float>(channel_name);

            for(size_t j = 0;j < W.rows();++j){
                c[j] = 0;
                for(size_t k = 0;k < duplicate;++k){
                    c[j] += W(j,i*duplicate + k);
                }
                // std::cout << W(j,0) << std::endl;
            }
        }
        set_output("mesh",mesh);
    }

    // decide the boundary points in V of skinned domain and evaluate their skinning weight
    bool boundary_conditions(
        const Eigen::MatrixXd& V,   /* The vertices of the skinned domain */
        const Eigen::MatrixXi& ,    /* Ele, but we don't need topology information here */
        const Eigen::MatrixXd& C,   /*  vertices of handles, include single point handles and bone edge handles' end points */
        const Eigen::VectorXi& P,   /* single point handles' indices */
        const Eigen::MatrixXi& BE,  /* bone edge handles' indices in pair */
        const Eigen::MatrixXi& CE,  /* cage edge handles' indices in pair */
        Eigen::VectorXi& b,
        Eigen::MatrixXd& bc,
        float single_point_influenced_radius,     /* influence radius of single point handles */
        float bone_edge_influenced_radius,        /* influence radius of bone edge handles */
        float cage_edge_influenced_radius         /* influence radius of cage_edge handles */
    ){
        using namespace Eigen;
        using namespace std;

        if(P.size() + BE.rows() == 0){
            std::cerr << "NO HANDLES FOUND" << std::endl;
            throw std::runtime_error("NO HANDLES FOUND");
        }

        // the sparse structure of bc matrix representing the weight matrix between domain vertices and handles
        vector<int> bci;
        vector<int> bcj;
        vector<double> bcv;

        // loop over single point handles, could be speedup using spacial data structure
        for(size_t p = 0;p < P.size();p++) {
            VectorXd pos = C.row(P[p]);
            // loop over skinned domain vertices
            for(size_t i = 0;i < V.rows();++i){
                VectorXd vi = V.row(i);
                double sqrd = (vi - pos).norm();
                if(sqrd <= single_point_influenced_radius){
                    bci.push_back(i);
                    bcj.push_back(p);
                    bcv.push_back(1.0);
                }
            }
        }

        // loop over bone edges
        for(size_t e = 0;e < BE.rows();++e){
            for(size_t i = 0;i < V.rows();++i){
                VectorXd tip =  C.row(BE(e,0));
                VectorXd tail = C.row(BE(e,1));
                double t,sqrd;
                igl::project_to_line(
                    V(i,0),V(i,1),V(i,2),
                    tip(0),tip(1),tip(2),
                    tail(0),tail(1),tail(2),
                    t,sqrd);
                
                if(t >= -bone_edge_influenced_radius && 
                    t <= (1.0 + bone_edge_influenced_radius) &&
                    std::sqrt(sqrd) <= bone_edge_influenced_radius){
                        bci.push_back(i);
                        bcj.push_back(P.size() + e);
                        bcv.push_back(1.0);
                }
            }
        }

        // loop over cage edges
        for(size_t e = 0;e < CE.rows();++e){
            for(size_t i = 0;i < V.rows();++i){
                VectorXd tip = C.row(CE(e,0));
                VectorXd tail = C.row(CE(e,1));

                double t,sqrd;
                igl::project_to_line(
                    V(i,0),V(i,1),V(i,2),
                    tip(0),tip(1),tip(2),
                    tail(0),tail(1),tail(2),
                    t,sqrd);
                if(t>=-cage_edge_influenced_radius &&
                        t <= (1.0f + cage_edge_influenced_radius) &&
                        std::sqrt(sqrd) <= cage_edge_influenced_radius)
                {
                    bci.push_back(i);
                    bcj.push_back(CE(e,0));
                    bcv.push_back(1.0-t);
                    bci.push_back(i);
                    bcj.push_back(CE(e,1));
                    bcv.push_back(t);
                }
            }
        }

        // find unique boundary indices
        vector<int> vb = bci;
        sort(vb.begin(),vb.end());
        vb.erase(unique(vb.begin(), vb.end()), vb.end());

        b.resize(vb.size());
        bc = MatrixXd::Zero(vb.size(),P.size()+BE.rows());
        // Map from boundary index to index in boundary
        map<int,int> bim;
        int i = 0;
        // Also fill in b
        for(vector<int>::iterator bit = vb.begin();bit != vb.end();bit++)
        {
            b(i) = *bit;
            bim[*bit] = i;
            i++;
        }

        // Build BC
        for(i = 0;i < (int)bci.size();i++)
        {
            assert(bim.find(bci[i]) != bim.end());
            bc(bim[bci[i]],bcj[i]) = bcv[i];
        }

        // Normalize across rows so that conditions sum to one
        for(i = 0;i<bc.rows();i++)
        {
            double sum = bc.row(i).sum();
            assert(sum != 0 && "Some boundary vertex getting all zero BCs");
            bc.row(i).array() /= sum;
        }

        if(bc.size() == 0)
        {
            // verbose("^%s: Error: boundary conditions are empty.\n",__FUNCTION__);
            return false;
        }

        // If there's only a single boundary condition, the following tests
        // are overzealous.
        if(bc.cols() == 1)
        {
            // If there is only one weight function,
            // then we expect that there is only one handle.
            assert(P.rows() + BE.rows() == 1);
            return true;
        }

        // Check that every Weight function has at least one boundary value of 1 and
        // one value of 0
        for(i = 0;i<bc.cols();i++)
        {
            double min_abs_c = bc.col(i).array().abs().minCoeff();
            double max_c = bc.col(i).maxCoeff();
            if(min_abs_c > 1e-6)
            {
            // verbose("^%s: Error: handle %d does not receive 0 weight\n",__FUNCTION__,i);
            return false;
            }
            if(max_c< (1-1e-6))
            {
            // verbose("^%s: Error: handle %d does not receive 1 weight\n",__FUNCTION__,i);
            return false;
            }
        }

        return true;


    }
};

ZENDEFNODE(GenerateSkinningWeight, {
    {"skinMesh","skinBone"},
    {"mesh"},
    {
        {"string","attr_prefix","sw"},
        {"float","sp_radius","1e-6"},
        {"float","bone_radius","1e-6"},
        {"float","cage_radius","1e-6"},
        {"float","duplicate","1.0"}
    },
    {"Skinning"},
});

};