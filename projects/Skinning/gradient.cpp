#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/grad.h>

namespace{
using namespace zeno;

struct EvalGradientOnPrimAttr : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("primIn");
        auto attrName = get_param<std::string>("attr_name");
        if(!prim->has_attr(attrName)){
            throw std::runtime_error("THE INPUT PRIMITIVE DOES NOT HAVE THE SPECIFIED SCALER FIELD");
        }

        Eigen::MatrixXd V;
        Eigen::MatrixXi E;
        V.resize(prim->size(),3);
        for(size_t i = 0;i < prim->size();++i)
            V.row(i) << prim->verts[i][0],prim->verts[i][1],prim->verts[i][2];
        if(prim->quads.size() == 0 && prim->tris.size() > 0){ // triangle surface mesh
            E.resize(prim->tris.size(),3);
            for(size_t i = 0;i < prim->tris.size();++i)
                E.row(i) << prim->tris[i][0],prim->tris[i][1],prim->tris[i][2];
        }else if(prim->quads.size() > 0){   // tetrahedron mesh
            E.resize(prim->quads.size(),4);
            for(size_t i = 0;i < prim->quads.size();++i)
                E.row(i) << prim->quads[i][0],prim->quads[i][1],prim->quads[i][2],prim->quads[i][3];
        }else{
            throw std::runtime_error("NO TOPOLOGY INFORMATION DETECTED IN LAPLACE SOLVER");
        }

        Eigen::VectorXd U;
        U.resize(prim->size());
        for(size_t i = 0;i < prim->size();++i)
            U[i] = prim->attr<float>(attrName)[i];

        Eigen::SparseMatrix<double> G;
        igl::grad(V,E,G);

        Eigen::MatrixXd e_grads = Eigen::Map<const Eigen::MatrixXd>((G*U).eval().data(),E.rows(),3);    

        std::vector<float> srs;
        srs.resize(prim->size(),0);
        std::vector<Eigen::Vector3d> p_grads;
        p_grads.resize(prim->size(),Eigen::Vector3d::Zero());
        for(size_t i = 0;i < prim->quads.size();++i){
            auto tet = prim->quads[i];
            Eigen::Vector3d e_grad = e_grads.row(i); 
            for(size_t k = 0;k < 4;++k){
                size_t l = (k+1) % 4;
                size_t m = (k+2) % 4;
                size_t n = (k+3) % 4;

                auto v0 = prim->verts[tet[k]];
                auto v1 = prim->verts[tet[l]];
                auto v2 = prim->verts[tet[m]];
                auto v3 = prim->verts[tet[n]];

                auto v10 = v1 - v0;
                auto v20 = v2 - v0;
                auto v30 = v3 - v0;

                auto l10 = zeno::length(v10);
                auto l20 = zeno::length(v20);
                auto l30 = zeno::length(v30);

                auto alpha = zeno::acos(zeno::dot(v10,v20)/l10/l20);
                auto beta = zeno::acos(zeno::dot(v10,v30)/l10/l30);
                auto gamma = zeno::acos(zeno::dot(v20,v30)/l20/l30);

                auto s = 0.5 * (alpha + beta + gamma);

                auto omega = 4*zeno::atan(zeno::sqrt(zeno::tan(s/2)*zeno::tan((s - alpha)/2)*zeno::tan((s-beta)/2)*zeno::tan((s-gamma)/2)));

                p_grads[tet[k]] += omega * e_grad;
                srs[tet[k]] += omega;
            }
        }    
        for(size_t i = 0;i < prim->size();++i)
            p_grads[i] /= srs[i];

        auto gradAttrName = attrName + "_grad";
        auto& gattr = prim->add_attr<zeno::vec3f>(gradAttrName);
        // gattr.resize(prim->size());
        for(size_t i = 0;i < prim->size();++i){
            gattr[i] = zeno::vec3f(p_grads[i][0],p_grads[i][1],p_grads[i][2]);
            // std::cout << gradAttrName << "<" << i << ">\t:" << p_grads[i].transpose() << std::endl;
        }
        set_output("primOut",prim);
    }
};

ZENDEFNODE(EvalGradientOnPrimAttr, {
    {"primIn"},
    {"primOut"},
    {{"string","attr_name","RENAME_ME"}},
    {"Skinning"},
});


};