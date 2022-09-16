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
        igl::copyleft::comiso::nrosy(V, F, b, bc, Eigen::VectorXi(), Eigen::VectorXd(), Eigen::MatrixXd(), 4, 0.5, X1, S);

        // Find the orthogonal vector
        Eigen::MatrixXd B1, B2, B3;
        igl::local_basis(V, F, B1, B2, B3);
        Eigen::MatrixXd X2 = igl::rotate_vectors(X1, Eigen::VectorXd::Constant(1, igl::PI / 2), B1, B2);

        Eigen::MatrixXd BIS1,BIS2;
        // Always work on the bisectors, it is more general
        igl::compute_frame_field_bisectors(V, F, X1, X2, BIS1, BIS2);

        // Comb the field, implicitly defining the seams
        Eigen::MatrixXd BIS1_combed, BIS2_combed;
        igl::comb_cross_field(V, F, BIS1, BIS2, BIS1_combed, BIS2_combed);

        // Find the integer mismatches
        Eigen::Matrix<int, Eigen::Dynamic, 3> MMatch;
        igl::cross_field_mismatch(V, F, BIS1_combed, BIS2_combed, true, MMatch);

        // Find the singularities
        Eigen::Matrix<int, Eigen::Dynamic, 1> isSingularity, singularityIndex;
        igl::find_cross_field_singularities(V, F, MMatch, isSingularity, singularityIndex);

        // Cut the mesh, duplicating all vertices on the seams
        Eigen::Matrix<int, Eigen::Dynamic, 3> Seams;
        igl::cut_mesh_from_singularities(V, F, MMatch, Seams);

        // Comb the frame-field accordingly
        // Combed field
        Eigen::MatrixXd X1_combed, X2_combed;
        igl::comb_frame_field(V, F, X1, X2, BIS1_combed, BIS2_combed, X1_combed, X2_combed);

        // Global parametrization
        Eigen::MatrixXd UV;
        Eigen::MatrixXi FUV;

        // Global parametrization
        igl::copyleft::comiso::miq(V,
                F,
                X1_combed,
                X2_combed,
                MMatch,
                isSingularity,
                Seams,
                UV,
                FUV,
                gradient_size,
                stiffness,
                direct_round,
                iter,
                5,
                true);

        // Global parametrization (with seams)
        Eigen::MatrixXd UV_seams;
        Eigen::MatrixXi FUV_seams;

        // Global parametrization (with seams, only for demonstration)
        igl::copyleft::comiso::miq(V,
                F,
                X1_combed,
                X2_combed,
                MMatch,
                isSingularity,
                Seams,
                UV_seams,
                FUV_seams,
                gradient_size,
                stiffness,
                direct_round,
                iter,
                5,
                false);


        // output the nrosy vector field, .... <B,B + global_scale * X1>
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
        auto prim_cross_field = std::make_shared<zeno::PrimitiveObject>();
        prim_cross_field->resize(B.rows() * 3);
        prim_cross_field->resize(B.rows() * 2);

        auto& verts_cross_field = prim_cross_field->verts;
        auto& lines_cross_field = prim_cross_field->lines;

        for(size_t i = 0;i < B.rows();++i) {
            verts_cross_field[i * 3 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            verts_cross_field[i * 3 + 1]  = verts_cross_field[i * 3 + 0] + zeno::vec3f(X1.row(i)[0],X1.row(i)[1],X1.row(i)[2]);
            verts_cross_field[i * 3 + 2]  = verts_cross_field[i * 3 + 0] + zeno::vec3f(X2.row(i)[0],X2.row(i)[1],X2.row(i)[2]);

            lines_cross_field[i*2 + 0] = zeno::vec2i(i * 3 + 0,i * 3 + 1);
            lines_cross_field[i*2 + 1] = zeno::vec2i(i * 3 + 0,i * 3 + 2);
        }       

        // output the bisector field
        auto prim_bisector_field = std::make_shared<zeno::PrimitiveObject>();
        prim_bisector_field->resize(B.rows() * 3);
        prim_bisector_field->resize(B.rows() * 2);  

        auto& verts_bisector_field = prim_bisector_field->verts;      
        auto& lines_bisector_field = prim_bisector_field->lines;  

        for(size_t i = 0;i < B.rows();++i) {
            verts_bisector_field[i * 3 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            verts_bisector_field[i * 3 + 1]  = verts_bisector_field[i * 3 + 0] + zeno::vec3f(BIS1.row(i)[0],BIS1.row(i)[1],BIS1.row(i)[2]);
            verts_bisector_field[i * 3 + 2]  = verts_bisector_field[i * 3 + 0] + zeno::vec3f(BIS2.row(i)[0],BIS2.row(i)[1],BIS2.row(i)[2]);

            lines_bisector_field[i*2 + 0] = zeno::vec2i(i * 3 + 0,i * 3 + 1);
            lines_bisector_field[i*2 + 1] = zeno::vec2i(i * 3 + 0,i * 3 + 2);
        }       

        // output bisector field combed
        auto prim_bisector_combed = std::make_shared<zeno::PrimitiveObject>();
        prim_bisector_combed->resize(B.rows() * 3);
        prim_bisector_combed->resize(B.rows() * 2);  

        auto& verts_bisector_combed = prim_bisector_combed->verts;      
        auto& lines_bisector_combed = prim_bisector_combed->lines;  

        for(size_t i = 0;i < B.rows();++i) {
            verts_bisector_combed[i * 3 + 0] = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            verts_bisector_combed[i * 3 + 1]  = verts_bisector_combed[i * 3 + 0] + zeno::vec3f(BIS1_combed.row(i)[0],BIS1_combed.row(i)[1],BIS1_combed.row(i)[2]);
            verts_bisector_combed[i * 3 + 2]  = verts_bisector_combed[i * 3 + 0] + zeno::vec3f(BIS2_combed.row(i)[0],BIS2_combed.row(i)[1],BIS2_combed.row(i)[2]);

            lines_bisector_combed[i*2 + 0] = zeno::vec2i(i * 3 + 0,i * 3 + 1);
            lines_bisector_combed[i*2 + 1] = zeno::vec2i(i * 3 + 0,i * 3 + 2);
        }

        // output singularities
        auto prim_singular_points = std::make_shared<zeno::PrimitiveObject>();
        prim_singular_points->resize(singularityIndex.size());
        for(size_t i = 0;i < singularityIndex.size();++i){
            prim_singular_points->verts[i] = prim->verts[singularityIndex[i]];
        }
        // output the seams
        int l_count = Seams.sum();
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

        auto prim_seams = std::make_shared<zeno::PrimitiveObject>();
        prim_seams->resize(l_count * 2);
        prim_seams->resize(l_count);
        auto& verts_seams = prim_seams->verts;
        auto& lines_seams = prim_seams->lines;
        for(unsigned int i = 0;i < l_count;++i){
            verts_seams[i * 2 + 0] = zeno::vec3f(P1.row(i)[0],P1.row(i)[1],P1.row(i)[2]);
            verts_seams[i * 2 + 1] = zeno::vec3f(P2.row(i)[0],P2.row(i)[1],P2.row(i)[2]);
            lines_seams[i] = zeno::vec2i(i * 2 + 0,i * 2 + 1);
        }

        // output the uv
        // auto& uv = prim->add_attr<zeno::vec3f>("uv");
        // for(size_t i = 0;size_t i < uv.size();++i)
        std::cout << "UV : " << UV.rows() << "\t" << UV.cols() << std::endl;
        std::cout << "FUC : " << FUV.rows() << "\t" << FUV.cols() << std::endl;
        std::cout << "V : " << V.rows() << "\t" << V.cols() << std::endl;

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