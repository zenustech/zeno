#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/local_basis.h>
#include <igl/copyleft/comiso/nrosy.h>
#include <igl/PI.h>
#include <igl/readOFF.h>

#include <iostream>

namespace {
using namespace zeno;

struct PrimitiveCalcTangentField : zeno::INode {

    void representative_to_nrosy(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXd& R,
        const int N,
        Eigen::MatrixXd& Y)
    {
        using namespace Eigen;
        using namespace std;
        MatrixXd B1, B2, B3;
        
        igl::local_basis(V,F,B1,B2,B3);
        
        Y.resize(F.rows()*N,3);
        for (unsigned i=0;i<F.rows();++i)
        {
            double x = R.row(i) * B1.row(i).transpose();
            double y = R.row(i) * B2.row(i).transpose();
            double angle = atan2(y,x);

            for (unsigned j=0; j<N;++j)
            {
                double anglej = angle + 2*igl::PI*double(j)/double(N);
                double xj = cos(anglej);
                double yj = sin(anglej);
                Y.row(i*N+j) = xj * B1.row(i) + yj * B2.row(i);

                if(fabs(Y.row(i*N + j).norm() - 1) > 0.5)
                    std::cout << "INVALID Y : " << i << "\t" << j << "\t" << Y.row(i*N + j) << std::endl;
            }
        }
    }

    virtual void apply() override {
        // using std;
        // using Eigen;

        #if 1

        auto prim = get_input<zeno::PrimitiveObject>("tris");
        auto cons_facet_tag = get_param<std::string>("cons_facet_tag");
        auto N = get_input2<int>("degree");
        // int N = 4;
        
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
        
        #else
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;

            igl::readOFF("/home/lsl/Project/libigl/build/_deps/libigl_tutorial_tata-src/bumpy.off", V, F);

            auto N = get_input2<int>("degree");
            int nm_tris = F.rows();
            int nm_vertices = V.rows();

        #endif

        // std::cout << "F : " << F.minCoeff() << std::endl;


        Eigen::VectorXi b;
        Eigen::MatrixXd bc;
        // int nm_constrained_facets = 0;
        // for(int i = 0;i < prim->tris.size();++i){
        //     auto tag = prim->tris.attr<float>(cons_facet_tag)[i];
        //     if(tag > 0) {
        //         nm_constrained_facets++;
        //     }
        // }

        // b.resize(nm_constrained_facets);
        // bc.resize(nm_constrained_facets,3);

        // int nm_constraine_facets = 1;
        b.resize(1);
        b << 0;
        bc.resize(1,3);
        bc << 1,1,1;


        // std::cout << "check here 0" << std::endl;

        Eigen::MatrixXd R;
        Eigen::VectorXd S;

        igl::copyleft::comiso::nrosy(V,F,b,bc,Eigen::VectorXi(),Eigen::VectorXd(),Eigen::MatrixXd(),N,0.5,R,S);

        // std::cout << "check here 1" << std::endl;

        double avg = igl::avg_edge_length(V,F);
        // std::cout << "avg : " << avg << std::endl;

        Eigen::MatrixXd Y;
        representative_to_nrosy(V,F,R,N,Y);

        // std::cout << "check here 2" << std::endl;

        Eigen::MatrixXd B;
        igl::barycenter(V,F,B);

        // std::cout << "check here 3" << std::endl;

        // Eigen::MatrixXd Be(B.rows()*N,3);
        // for(unsigned i=0; i<B.rows();++i)
        //     for(unsigned j=0; j<N; ++j)
        //     Be.row(i*N+j) = B.row(i);


        // Eigen::MatrixXd Be(B.rows()*N,3);
        // for(unsigned i=0; i<B.rows();++i)
        //     for(unsigned j=0; j<N; ++j)
        //         Be.row(i*N+j) = B.row(i);

        // output the nrosy field
        auto nrosy_fields = std::make_shared<zeno::PrimitiveObject>();

        nrosy_fields->verts.resize(nm_tris * (N + 1));
        nrosy_fields->lines.resize(nm_tris * N);

        auto& fverts = nrosy_fields->verts;
        auto& flines = nrosy_fields->lines;
        
        // std::cout << "Y : " << Y.rows() << "\t" << Y.cols() << std::endl;
        // std::cout << "F : " << F.rows() << "\t" << F.cols() << std::endl;

        for(int i = 0;i < nm_tris;++i){
            auto Bv = zeno::vec3f(B.row(i)[0],B.row(i)[1],B.row(i)[2]);
            fverts[i * (N + 1) + 0] = Bv;
            for(int j = 0;j < N;++j){
                auto Yv = zeno::vec3f(Y.row(i*N + j)[0],Y.row(i*N + j)[1],Y.row(i*N + j)[2]);
                fverts[i * (N + 1) + j + 1] = Bv + Yv * avg / 2;
                flines[i * N + j] = zeno::vec2i(i * (N + 1) + 0,i * (N + 1) + j + 1);// add lines
            }
        }


        // output the singularity points
        auto singular_points = std::make_shared<zeno::PrimitiveObject>();

        int nm_singular_points = 0;

        for(size_t i = 0;i < S.size();++i){
            if(S[i] < -1e-6){
                singular_points->verts.emplace_back(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
                ++nm_singular_points;
            }
        }
        std::cout << "nm_singular_points : " << nm_singular_points << std::endl;
        singular_points->resize(singular_points->size());



        // output the surface mesh
        auto primOut = std::make_shared<zeno::PrimitiveObject>();
        primOut->verts.resize(V.rows());
        primOut->tris.resize(F.rows());

        for(int i = 0;i < V.rows();++i)
            primOut->verts[i] = zeno::vec3f(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
        for(int i = 0;i < F.rows();++i)
            primOut->tris[i] = zeno::vec3i(F.row(i)[0],F.row(i)[1],F.row(i)[2]);

        set_output("nrosy_fields",std::move(nrosy_fields));
        set_output("singular_points",std::move(singular_points));
        set_output("trisOut",std::move(primOut));
    }
};

ZENDEFNODE(PrimitiveCalcTangentField, {
    {"tris",{"int","degree","4"}},
    {"nrosy_fields","singular_points","trisOut"},
    {{"string","cons_facet_tag","RENAME_ME"}},
    {"Parameterization"},
});


};