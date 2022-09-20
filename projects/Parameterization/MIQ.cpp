#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/comb_cross_field.h>
#include <igl/comb_frame_field.h>
#include <igl/compute_frame_field_bisectors.h>
#include <igl/cross_field_mismatch.h>
#include <igl/cut_mesh_from_singularities.h>
#include <igl/find_cross_field_singularities.h>
#include <igl/local_basis.h>
#include <igl/readOFF.h>
#include <igl/rotate_vectors.h>
#include <igl/copyleft/comiso/miq.h>
#include <igl/copyleft/comiso/nrosy.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/PI.h>

#include <iostream>

namespace {

using namespace zeno;

struct PrimitiveCalcUVMapMIQ : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");

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


        double gradient_size = 50;
        double iter = 0;
        double stiffness = 5.0;
        bool direct_round = 0;

        Eigen::MatrixXd B;
        // Compute face barycenters
        igl::barycenter(V, F, B);

        // Compute scale for visualizing fields
        double global_scale =  .5*igl::avg_edge_length(V, F);

        // Contrain one face
        Eigen::VectorXi b(1);
        b << 0;
        Eigen::MatrixXd bc(1, 3);
        bc << 1, 0, 0;

        // Create a smooth 4-RoSy field
        Eigen::VectorXd S;
        Eigen::MatrixXd X1;
        std::cout << "compute nrosy" << std::endl;
        igl::copyleft::comiso::nrosy(V, F, b, bc, Eigen::VectorXi(), Eigen::VectorXd(), Eigen::MatrixXd(), 4, 0.5, X1, S);

        // Find the orthogonal vector
        Eigen::MatrixXd B1, B2, B3;
        igl::local_basis(V, F, B1, B2, B3);
        Eigen::MatrixXd X2 = igl::rotate_vectors(X1, Eigen::VectorXd::Constant(1, igl::PI / 2), B1, B2);

        Eigen::MatrixXd BIS1,BIS2;
        // Always work on the bisectors, it is more general
        std::cout << "compute frame field bisectors" << std::endl;
        igl::compute_frame_field_bisectors(V, F, X1, X2, BIS1, BIS2);
        std::cout << "check BIS1 : " << BIS1.row(283) << BIS1.row(284) << std::endl;
        std::cout << "check BIS2 : " << BIS2.row(283) << BIS2.row(284) << std::endl;

        // Comb the field, implicitly defining the seams
        std::cout << "compute cross field" << std::endl;
        Eigen::MatrixXd BIS1_combed, BIS2_combed;
        igl::comb_cross_field(V, F, BIS1, BIS2, BIS1_combed, BIS2_combed);

        // Find the integer mismatches
        std::cout << "cross field mismatch" << std::endl;
        Eigen::Matrix<int, Eigen::Dynamic, 3> MMatch;
        std::cout << "check BIS1 comb: " << BIS1_combed.row(283) << BIS1_combed.row(284) << std::endl;
        std::cout << "check BIS2 comb: " << BIS2_combed.row(283) << BIS2_combed.row(284) << std::endl;
        igl::cross_field_mismatch(V, F, BIS1_combed, BIS2_combed, true, MMatch);

        // Find the singularities
        std::cout << "find cross field singularities" << std::endl;
        Eigen::Matrix<int, Eigen::Dynamic, 1> isSingularity, singularityIndex;
        igl::find_cross_field_singularities(V, F, MMatch, isSingularity, singularityIndex);

        // Cut the mesh, duplicating all vertices on the seams
        std::cout << "cut mesh from singularities" << std::endl;
        Eigen::Matrix<int, Eigen::Dynamic, 3> Seams;
        std::cout << "MMatch : " << MMatch.sum() << std::endl;
        std::cout << "NM_SINGULAR : " << isSingularity.sum() << std::endl;
        igl::cut_mesh_from_singularities(V, F, MMatch, Seams);

        std::cout << "output seams" << std::endl;
        int l_count = Seams.sum();

        // Comb the frame-field accordingly
        // Combed field
        std::cout << "compute frame field" << std::endl;
        Eigen::MatrixXd X1_combed, X2_combed;
        igl::comb_frame_field(V, F, X1, X2, BIS1_combed, BIS2_combed, X1_combed, X2_combed);
        std::cout << "finish compute frame field" << std::endl;

        // // Global parametrization
        // Eigen::MatrixXd UV;
        // Eigen::MatrixXi FUV;

        // std::cout << "solve for global parameterization" << std::endl;
        // // Global parametrization
        // igl::copyleft::comiso::miq(V,
        //         F,
        //         X1_combed,
        //         X2_combed,
        //         MMatch,
        //         isSingularity,
        //         Seams,
        //         UV,
        //         FUV,
        //         gradient_size,
        //         stiffness,
        //         direct_round,
        //         iter,
        //         5,
        //         true);

        // // Global parametrization (with seams)
        // Eigen::MatrixXd UV_seams;
        // Eigen::MatrixXi FUV_seams;

        // // Global parametrization (with seams, only for demonstration)
        // std::cout << "solve for global param with seams " << std::endl;
        // igl::copyleft::comiso::miq(V,
        //         F,
        //         X1_combed,
        //         X2_combed,
        //         MMatch,
        //         isSingularity,
        //         Seams,
        //         UV_seams,
        //         FUV_seams,
        //         gradient_size,
        //         stiffness,
        //         direct_round,
        //         iter,
        //         5,
        //         false);


        // output the nrosy vector field, .... <B,B + global_scale * X1>
        std::cout << "output nrosy" << std::endl;
        auto prim_nrosy = std::make_shared<zeno::PrimitiveObject>();
        prim_nrosy->resize(B.rows() * 2);
        prim_nrosy->lines.resize(B.rows());

        auto& verts_nrosy = prim_nrosy->verts;
        auto& lines_nrosy = prim_nrosy->lines;
        for(size_t i = 0;i < B.rows();++i){
            verts_nrosy[i * 2 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            verts_nrosy[i * 2 + 1] = verts_nrosy[i * 2 + 0] + global_scale * zeno::vec3f(X1.row(i)[0],X1.row(i)[1],X1.row(i)[2]);
            lines_nrosy[i] = zeno::vec2i(i * 2 + 0,i * 2 + 1);
        }

        // output the the cross field
        std::cout << "output cross field" << std::endl;
        auto prim_cross_field = std::make_shared<zeno::PrimitiveObject>();
        prim_cross_field->resize(B.rows() * 4);
        prim_cross_field->lines.resize(B.rows() * 2);

        auto& verts_cross_field = prim_cross_field->verts;
        auto& lines_cross_field = prim_cross_field->lines;
        auto& clrs_cross_field = prim_cross_field->add_attr<zeno::vec3f>("clr");

        for(size_t i = 0;i < B.rows();++i) {
            verts_cross_field[i * 4 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_cross_field[i * 4 + 0] = zeno::vec3f(1,0,0);
            verts_cross_field[i * 4 + 1]  = verts_cross_field[i * 4 + 0] + global_scale * zeno::vec3f(X1.row(i)[0],X1.row(i)[1],X1.row(i)[2]);
            clrs_cross_field[i * 4 + 1] = zeno::vec3f(1,0,0);
            verts_cross_field[i * 4 + 2] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_cross_field[i * 4 + 2] = zeno::vec3f(0,0,1);
            verts_cross_field[i * 4 + 3]  = verts_cross_field[i * 4 + 2] + global_scale * zeno::vec3f(X2.row(i)[0],X2.row(i)[1],X2.row(i)[2]);
            clrs_cross_field[i * 4 + 3] = zeno::vec3f(0,0,1);

            lines_cross_field[i*2 + 0] = zeno::vec2i(i * 4 + 0,i * 4 + 1);
            lines_cross_field[i*2 + 1] = zeno::vec2i(i * 4 + 2,i * 4 + 3);
        }       

        // output the bisector field
        std::cout << "output cross field bisector" << std::endl;
        auto prim_bisector_field = std::make_shared<zeno::PrimitiveObject>();
        prim_bisector_field->resize(B.rows() * 4);
        prim_bisector_field->lines.resize(B.rows() * 2);  

        auto& verts_bisector_field = prim_bisector_field->verts;      
        auto& lines_bisector_field = prim_bisector_field->lines;  
        auto& clrs_bisector_field = prim_bisector_field->add_attr<zeno::vec3f>("clr");
        

        for(size_t i = 0;i < B.rows();++i) {
            verts_bisector_field[i * 4 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_bisector_field[i * 4 + 0] = zeno::vec3f(1,0,0);
            verts_bisector_field[i * 4 + 1]  = verts_bisector_field[i * 4 + 0] + global_scale * zeno::vec3f(BIS1.row(i)[0],BIS1.row(i)[1],BIS1.row(i)[2]);
            clrs_bisector_field[i * 4 + 1] = zeno::vec3f(1,0,0);
            verts_bisector_field[i * 4 + 2] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_bisector_field[i * 4 + 2] = zeno::vec3f(0,0,1);
            verts_bisector_field[i * 4 + 3]  = verts_bisector_field[i * 4 + 2] + global_scale * zeno::vec3f(BIS2.row(i)[0],BIS2.row(i)[1],BIS2.row(i)[2]);
            clrs_bisector_field[i * 4 + 3] = zeno::vec3f(0,0,1);

            lines_bisector_field[i*2 + 0] = zeno::vec2i(i * 4 + 0,i * 4 + 1);
            lines_bisector_field[i*2 + 1] = zeno::vec2i(i * 4 + 2,i * 4 + 3);
        }       

        // output bisector field combed
        std::cout << "output cross field bisector combed" << std::endl;
        auto prim_bisector_combed = std::make_shared<zeno::PrimitiveObject>();
        prim_bisector_combed->resize(B.rows() * 4);
        prim_bisector_combed->lines.resize(B.rows() * 2);  

        auto& verts_bisector_combed = prim_bisector_combed->verts;      
        auto& lines_bisector_combed = prim_bisector_combed->lines;  
        auto& clrs_bisector_combed = prim_bisector_combed->add_attr<zeno::vec3f>("clr");

        for(size_t i = 0;i < B.rows();++i) {
            verts_bisector_combed[i * 4 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_bisector_combed[i * 4 + 0] = zeno::vec3f(1,0,0);
            verts_bisector_combed[i * 4 + 1]  = verts_bisector_combed[i * 4 + 0] + global_scale * zeno::vec3f(BIS1_combed.row(i)[0],BIS1_combed.row(i)[1],BIS1_combed.row(i)[2]);
            clrs_bisector_combed[i * 4 + 1] = zeno::vec3f(1,0,0);
            verts_bisector_combed[i * 4 + 2] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            clrs_bisector_combed[i * 4 + 2] = zeno::vec3f(0,0,1);
            verts_bisector_combed[i * 4 + 3]  = verts_bisector_combed[i * 4 + 2] + global_scale * zeno::vec3f(BIS2_combed.row(i)[0],BIS2_combed.row(i)[1],BIS2_combed.row(i)[2]);
            clrs_bisector_combed[i * 4 + 3] = zeno::vec3f(0,0,1);

            lines_bisector_combed[i*2 + 0] = zeno::vec2i(i * 4 + 0,i * 4 + 1);
            lines_bisector_combed[i*2 + 1] = zeno::vec2i(i * 4 + 2,i * 4 + 3);
        }

        // output singularities
        std::cout << "output singularities" << std::endl;
        auto prim_singular_points = std::make_shared<zeno::PrimitiveObject>();
        int nm_singular_points = 0;
        for(int i = 0;i < singularityIndex.size();++i)
            if(singularityIndex[i] > 0)
                nm_singular_points++;

        prim_singular_points->resize(nm_singular_points);
        auto& rads_singular_points = prim_singular_points->add_attr<float>("rad");
        auto& clrs_singular_points = prim_singular_points->add_attr<zeno::vec3f>("clr");

        size_t sp_idx = 0;
        for(size_t i = 0;i < singularityIndex.size();++i){
            if(singularityIndex[i] > 0 && singularityIndex[i] < 2){
                prim_singular_points->verts[sp_idx] = prim->verts[i];
                rads_singular_points[sp_idx] = 10;
                clrs_singular_points[sp_idx] = zeno::vec3f(1,0,0);
                ++sp_idx;
            }
            if(singularityIndex[i] > 2){
                prim_singular_points->verts[sp_idx] = prim->verts[i];
                rads_singular_points[sp_idx] = 10;
                clrs_singular_points[sp_idx] = zeno::vec3f(0,1,0);
                ++sp_idx;
            }            
        }
        // output the seams
        Eigen::MatrixXd P1(l_count,3);
        Eigen::MatrixXd P2(l_count,3);

        for (unsigned i=0; i<Seams.rows(); ++i)
        {
            for (unsigned j=0; j<Seams.cols(); ++j)
            {
                if (Seams(i,j) != 0)
                {
                    P1.row(l_count-1) = V.row(F(i,j));
                    P2.row(l_count-1) = V.row(F(i,(j+1)%3));
                    l_count--;
                }
            }
        }

        std::cout << "l_count = " << l_count << std::endl;
        auto prim_seams = std::make_shared<zeno::PrimitiveObject>();
        prim_seams->resize(l_count * 2);
        prim_seams->lines.resize(l_count);
        auto& verts_seams = prim_seams->verts;
        auto& lines_seams = prim_seams->lines;
        for(unsigned int i = 0;i < l_count;++i){
            verts_seams[i * 2 + 0] = zeno::vec3f(P1.row(i)[0],P1.row(i)[1],P1.row(i)[2]);
            verts_seams[i * 2 + 1] = zeno::vec3f(P2.row(i)[0],P2.row(i)[1],P2.row(i)[2]);
            lines_seams[i] = zeno::vec2i(i * 2 + 0,i * 2 + 1);
        }

        std::cout << "finish output" << std::endl;

        // output the uv
        // auto& uv = prim->add_attr<zeno::vec3f>("uv");
        // for(size_t i = 0;size_t i < uv.size();++i)
        // std::cout << "UV : " << UV.rows() << "\t" << UV.cols() << std::endl;
        // std::cout << "FUC : " << FUV.rows() << "\t" << FUV.cols() << std::endl;
        // std::cout << "V : " << V.rows() << "\t" << V.cols() << std::endl;

         set_output("nrosy",std::move(prim_nrosy));
         set_output("cross_field",std::move(prim_cross_field));
         set_output("bisector_field",std::move(prim_bisector_field));
         set_output("bisector_combed",std::move(prim_bisector_combed));
         set_output("singular_points",std::move(prim_singular_points));
         set_output("seams",std::move(prim_seams));

    }
};

ZENDEFNODE(PrimitiveCalcUVMapMIQ, {
    {"prim"},
    {"nrosy","cross_field","bisector_field","bisector_combed","singular_points","seams"},
    {},
    {"Parameterization"},
});

};