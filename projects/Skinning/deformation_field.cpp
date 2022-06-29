#include "skinning_header.h"

#include <iostream>

namespace{
using namespace zeno;
// with relatively large rotation, directly interpolate on displacement field will be problematic, the skin apt to be shrinking
struct RetrieveDisplacementField : zeno::INode {
    virtual void apply() override {
        auto shape = get_input<zeno::PrimitiveObject>("shape");
        auto& disps = shape->add_attr<zeno::vec3f>("disp");
        shape->resize(shape->size());

        const auto& pos = shape->attr<zeno::vec3f>("pos");
        const auto& curPos = shape->attr<zeno::vec3f>("curPos");
        for(size_t i = 0;i < shape->size();++i){
            disps[i] = curPos[i] - pos[i];
        }

        set_output("shape",shape);      
    }
};
ZENDEFNODE(RetrieveDisplacementField,{
    {"shape"},
    {"shape"},
    {},
    {"Skinning"},
});
// The two input primitive objects should be volumetric mesh, currently only quads is supported
// Using linear blending for interpolating the nodal affine field from neighbored elements
struct RetrieveAffineField : zeno::INode {
    virtual void apply() override {
        // auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        // auto def_shape = get_input<zeno::PrimitiveObject>("def_shape");
        auto shape = get_input<zeno::PrimitiveObject>("shape");

        size_t nm_quads = shape->quads.size();

        std::vector<Eigen::Matrix4d> nodal_affine;
        nodal_affine.resize(shape->size(),Eigen::Matrix4d::Zero());
        std::vector<double> nodal_weight_sum;
        nodal_weight_sum.resize(shape->size(),0);

        const auto& pos = shape->attr<zeno::vec3f>("pos");
        const auto& curPos = shape->attr<zeno::vec3f>("curPos");

        for(size_t i = 0;i < nm_quads;++i){
            const auto& quad = shape->quads[i];
            Eigen::Matrix4d ref_pos;
            Eigen::Matrix4d def_pos;

            std::vector<Eigen::Vector3d> elm_verts(4);

            for(size_t j = 0;j < 4;++j){
                size_t idx = quad[j];
                elm_verts[j] << pos[idx][0],pos[idx][1],pos[idx][2];
                ref_pos.col(j) << pos[idx][0],pos[idx][1],pos[idx][2],1.0;
                def_pos.col(j) << curPos[idx][0],curPos[idx][1],curPos[idx][2],1.0;
            }

            Eigen::Matrix4d affine = def_pos * ref_pos.inverse();
            // using the distance from the point to the opposite triangle as the weight
            for(size_t j = 0;j < 4;++j){
                double w = Height(elm_verts[j],elm_verts[(j+1)%4],elm_verts[(j+2)%4],elm_verts[(j+3)%4]);
                nodal_affine[quad[j]] += w * affine;
                nodal_weight_sum[quad[j]] += w;
            }
        }


        for(size_t i = 0;i < nodal_affine.size();++i){
            nodal_affine[i] /= nodal_weight_sum[i];
            if(std::isnan(nodal_affine[i].norm())){
                std::cerr << "NAN AFFINE DETECTED : " << nodal_weight_sum[i] << std::endl << nodal_affine[i] << std::endl;
                throw std::runtime_error("NAN AFFINE RETRIEVAL");
            }
        }

        auto& affine0 = shape->add_attr<zeno::vec3f>("A0");
        auto& affine1 = shape->add_attr<zeno::vec3f>("A1");
        auto& affine2 = shape->add_attr<zeno::vec3f>("A2");
        auto& affine3 = shape->add_attr<zeno::vec3f>("A3");
        shape->resize(shape->size());

        for(size_t i = 0;i < shape->size();++i){
            affine0[i] = zeno::vec3f(nodal_affine[i](0,0),nodal_affine[i](0,1),nodal_affine[i](0,2));
            affine1[i] = zeno::vec3f(nodal_affine[i](1,0),nodal_affine[i](1,1),nodal_affine[i](1,2));
            affine2[i] = zeno::vec3f(nodal_affine[i](2,0),nodal_affine[i](2,1),nodal_affine[i](2,2));
            affine3[i] = zeno::vec3f(nodal_affine[i](0,3),nodal_affine[i](1,3),nodal_affine[i](2,3));
        }

        set_output("shape",shape); 
    }

    static double Height(const Eigen::Vector3d& v0,const Eigen::Vector3d& v1,const Eigen::Vector3d&v2,const Eigen::Vector3d& v3){
        Eigen::Vector3d v30 = v3 - v0;
        Eigen::Vector3d v20 = v2 - v0;
        Eigen::Vector3d v10 = v1 - v0;

        Eigen::Vector3d v10xv20 = v10.cross(v20);
        v10xv20 /= v10xv20.norm();

        return fabs(v30.dot(v10xv20));
    }

};
ZENDEFNODE(RetrieveAffineField,{
    {"shape"},
    {"shape"},
    {},
    {"Skinning"},
});
struct AlignPrimitive : zeno::INode {
    virtual void apply() override {
        auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        auto aligned_shape = get_input<zeno::PrimitiveObject>("aligned_shape");

        auto res = std::make_shared<zeno::PrimitiveObject>(*aligned_shape);

        const auto& rtris = ref_shape->tris;
        const auto& atris = aligned_shape->tris;

        const auto& rquads = ref_shape->quads;
        const auto& aquads = aligned_shape->quads;

        if(ref_shape->size() != aligned_shape->size()){
            throw std::runtime_error("AlignPrimitiveObject : INPUT SHAPES SIZE NOT MATCH");
        }

        if(rtris->size() != atris->size()){
            throw std::runtime_error("AlignPrimitiveObject : INPUT TRIS SIZE NOT MATCH");
        }

        if(rquads->size() != aquads->size()){
            throw std::runtime_error("AlignPrimitiveObject : INPUT QUADS SIZE NOT MATCH");
        }

        bool conflict_tris = false;
        int count = 0;
        for(size_t i = 0;i < rtris.size();++i){
            auto tri_diff = rtris[i] - atris[i];
            if(tri_diff[0] != 0 || tri_diff[1] != 0 || tri_diff[2] != 0){
                std::cout << "CONFLICT TRIS : " << i << std::endl;
                std::cout << "rtris : " << rtris[i][0] << "\t" << rtris[i][1] << "\t" << rtris[i][2] << std::endl;
                std::cout << "atris : " << atris[i][0] << "\t" << atris[i][1] << "\t" << atris[i][2] << std::endl;
                conflict_tris = true;
                count++;

                if(count > 20)
                    break;
                // throw std::runtime_error("AlignPrimitiveObject : INPUT TRIS TOPO NOT MATCH");
            }
        }

        // if(conflict_tris)
        //     throw std::runtime_error("AlignPrimitiveObject : INPUT TRIS TOPO NOT MATCH");

        for(size_t i = 0;i < rquads.size();++i){
            auto quad_diff = rquads[i] - aquads[i];
            if(quad_diff[0] != 0 || quad_diff[1] != 0 || quad_diff[2] != 0 || quad_diff[3] != 0){
                throw std::runtime_error("AlignPrimitiveObject : INPUT QUADS TOPO NOT MATCH");
            }
        }


        
        #pragma omp parallel for 
        for(size_t i = 0;i < res->size();++i){
            res->verts[i] = ref_shape->verts[i];
        }

        set_output("res",std::move(res));
    }
};
ZENDEFNODE(AlignPrimitive,{
    {{"ref_shape"},{"aligned_shape"}},
    {"res"},
    {},
    {"Skinning"},
});

struct EvalSurfaceDeformtion : zeno::INode {
    double computeArea(const zeno::vec3f& p0,
                        const zeno::vec3f& p1,
                        const zeno::vec3f& p2) const {
        auto p01 = p0 - p1;
        auto p02 = p0 - p2;
        auto p12 = p1 - p2;
        auto a = zeno::length(p01);
        auto b = zeno::length(p02);
        auto c = zeno::length(p12);

        auto s = (a + b + c)/2;
        return std::sqrt(s*(s-a)*(s-b)*(s-c));
    }

    virtual void apply() override {
        auto prim_ref = get_input<zeno::PrimitiveObject>("prim_ref");
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto outAttr = get_param<std::string>("attr");

        size_t nm_tris = prim_ref->tris.size();
        if(prim->tris.size() != nm_tris || prim->size() != prim_ref->size()){
            throw std::runtime_error("THE SIZE OF TWO INPUT SURFACE NOT MATCH");
        }

        for(size_t ti = 0;ti < nm_tris;++ti){
            const auto& tri_ref = prim_ref->tris[ti];
            const auto& tri = prim->tris[ti];

            if(tri_ref[0] != tri[0] || tri_ref[1] != tri[1] || tri_ref[2] != tri[2])
                throw std::runtime_error("THE TOPO OF TWO INPUT SURFACE NOT MATCH");
        }

        std::vector<int> nm_neigh_tris(prim->size());
        std::fill(nm_neigh_tris.begin(),nm_neigh_tris.end(),0);

        auto& outDefs = prim->add_attr<float>(outAttr);
        const auto& pos_ref = prim_ref->attr<zeno::vec3f>("pos");
        const auto& pos = prim->attr<zeno::vec3f>("pos");
        std::cout << "nm_tris:" << nm_tris << std::endl;
        for(size_t ti = 0;ti < nm_tris;++ti){
            const auto& tri = prim->tris[ti];
            auto Aref = computeArea(pos_ref[tri[0]],pos_ref[tri[1]],pos_ref[tri[2]]);
            auto A = computeArea(pos[tri[0]],pos[tri[1]],pos[tri[2]]);

            auto adef = A/Aref;
            nm_neigh_tris[tri[0]] += 1;
            nm_neigh_tris[tri[1]] += 1;
            nm_neigh_tris[tri[2]] += 1;

            outDefs[tri[0]] += adef;
            outDefs[tri[1]] += adef;
            outDefs[tri[2]] += adef;
        }

        for(size_t i = 0;i < prim->size();++i)
            outDefs[i] /= nm_neigh_tris[i];

        set_output("prim",prim);
    }
};

ZENDEFNODE(EvalSurfaceDeformtion,{
    {{"prim_ref"},{"prim"}},
    {"prim"},
    {{"string","attr","def"}},
    {"Skinning"},
});


};