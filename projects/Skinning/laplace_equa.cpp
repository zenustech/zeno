#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/sort.h>
// #include <igl/list_to_matrix.h>
#include <igl/slice.h>
#include <igl/setdiff.h>
#include <igl/slice_into.h>

namespace{
using namespace zeno;

struct SolveLaplaceEquaOnAttr : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("primIn");
        if(!prim->has_attr("btag")){
            throw std::runtime_error("FOR SOLVING LAPLACE EQUA THE BTAG ATTR SHOULD BE MARKED");
        }
        auto attr_name = get_param<std::string>("attr_name");
        if(!prim->has_attr(attr_name)){
            throw std::runtime_error("THE PRIM DOES NOT HAVE WANTED ATTR");
        }
        Eigen::VectorXd x;
        x.resize(prim->size());
        for(size_t i = 0;i < prim->size();++i)
            x[i] = prim->attr<float>(attr_name)[i];


        std::vector<size_t> closeBouIndices;
        std::vector<size_t> farBouIndices;
        const auto& btag = prim->attr<float>("btag");
        // for verts with btag == 1.0 are close-end boundary points and btag == 2.0 for far-end boundary points
        closeBouIndices.clear();
        farBouIndices.clear();
        for(size_t i = 0;i < prim->size();++i){
            if(btag[i] == 1.0)
                closeBouIndices.push_back(i);
            if(btag[i] == 2.0)
                farBouIndices.push_back(i);
        }

        size_t nm_cb = closeBouIndices.size();
        size_t nm_fb = farBouIndices.size();

        Eigen::MatrixXd V,E;
        V.resize(prim->size(),3);
        for(size_t i = 0;i < prim->size();++i)
            V.row(i) << prim->verts[i][0],prim->verts[i][1],prim->verts[i][2];
        if(prim->quads.size() == 0 && prim->tris.size() > 0){ // triangle surface mesh
            E.resize(prim->tris.size(),3);
            for(size_t i = 0;i < prim->tris.size();++i)
                E.row(i) << prim->tris[i][0],prim->tris[i][1],prim->tris[i][2];
        }else if(prim->quads.size() > 0){
            E.resize(prim->quads.size(),4);
            for(size_t i = 0;i < prim->quads.size();++i)
                E.row(i) << prim->quads[i][0],prim->quads[i][1],prim->quads[i][2],prim->quads[i][3];
        }else{
            throw std::runtime_error("NO TOPOLOGY INFORMATION DETECTED IN LAPLACE SOLVER");
        }

        Eigen::VectorXi all,b_unsorted,b,in;
        b.resize(nm_cb + nm_fb);
        b_unsorted.resize(nm_cb + nm_fb);
        for(size_t i = 0;i < nm_cb;++i)
            b_unsorted[i] = closeBouIndices[i];
        for(size_t i = 0;i < nm_fb;++i)
            b_unsorted[i + nm_cb] = farBouIndices[i];
        igl::sort(b_unsorted,1,true,b);

        igl::colon(0,prim->size()-1,all);
        Eigen::VectorXi AI;
        igl::setdiff(all,b,in,AI);

        Eigen::SparseMatrix<double> L,L_in_in,L_in_b;
        igl::cotmatrix(V,E,L);
        igl::slice(L,in,in,L_in_in);
        igl::slice(L,in,b,L_in_b);

        Eigen::VectorXd xb;
        igl::slice(x,b,xb);

        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(-L_in_in);
        Eigen::VectorXd xin = solver.solve(L_in_b*xb);

        igl::slice_into(xin,in,x);
        for(size_t i = 0;i < prim->size();++i)
            prim->attr<float>(attr_name)[i] = x[i];

        set_output("primOut",prim);
    }
};

ZENDEFNODE(SolveLaplaceEquaOnAttr, {
    {"primIn"},
    {"primOut"},
    {{"string","attr_name","RENAME_ME"}},
    {"Skinning"},
});

struct SolveLaplaceEquation : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("primIn");
        if(!prim->has_attr("btag")){
            throw std::runtime_error("FOR SOLVING LAPLACE EQUA THE BTAG ATTR SHOULD BE MARKED");
        }
        auto attr_name = get_param<std::string>("attr_name");
        if(!prim->has_attr(attr_name)){
            throw std::runtime_error("THE PRIM DOES NOT HAVE WANTED ATTR");
        }
        Eigen::VectorXd x;
        x.resize(prim->size());
        for(size_t i = 0;i < prim->size();++i)
            x[i] = prim->attr<float>(attr_name)[i];

        std::vector<size_t> bouIndices;
        const auto& btag = prim->attr<float>("btag");

        bouIndices.clear();
        for(size_t i = 0;i < prim->size();++i)
            if(btag[i] == 1.0)
                bouIndices.push_back(i);

        size_t nm_bp = bouIndices.size();

        Eigen::MatrixXd V,E;
        V.resize(prim->size(),3);
        for(size_t i = 0;i < prim->size();++i)
            V.row(i) << prim->verts[i][0],prim->verts[i][1],prim->verts[i][2];
        if(prim->quads.size() == 0 && prim->tris.size() > 0){ // triangle surface mesh
            E.resize(prim->tris.size(),3);
            for(size_t i = 0;i < prim->tris.size();++i)
                E.row(i) << prim->tris[i][0],prim->tris[i][1],prim->tris[i][2];
        }else if(prim->quads.size() > 0){
            E.resize(prim->quads.size(),4);
            for(size_t i = 0;i < prim->quads.size();++i)
                E.row(i) << prim->quads[i][0],prim->quads[i][1],prim->quads[i][2],prim->quads[i][3];
        }else{
            throw std::runtime_error("NO TOPOLOGY INFORMATION DETECTED IN LAPLACE SOLVER");
        }

        Eigen::VectorXi all,b_unsorted,b,in;
        b.resize(nm_bp);
        b_unsorted.resize(nm_bp);
        for(size_t i = 0;i < nm_bp;++i)
            b_unsorted[i] = bouIndices[i];
        igl::sort(b_unsorted,1,true,b);

        igl::colon(0,prim->size()-1,all);
        Eigen::VectorXi AI;
        igl::setdiff(all,b,in,AI);

        Eigen::SparseMatrix<double> L,L_in_in,L_in_b;
        igl::cotmatrix(V,E,L);
        igl::slice(L,in,in,L_in_in);
        igl::slice(L,in,b,L_in_b);

        Eigen::VectorXd xb;
        igl::slice(x,b,xb);

        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(-L_in_in);
        Eigen::VectorXd xin = solver.solve(L_in_b*xb);

        igl::slice_into(xin,in,x);
        for(size_t i = 0;i < prim->size();++i)
            prim->attr<float>(attr_name)[i] = x[i];

        set_output("primOut",prim);
    }
};

ZENDEFNODE(SolveLaplaceEquation, {
    {"primIn"},
    {"primOut"},
    {{"string","attr_name","RENAME_ME"}},
    {"Skinning"},
});



};