#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include "skinning_iobject.h"

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
    {"disp_field"},
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
        for(size_t i = 0;i < res->size();++i)
            res->verts[i] = ref_shape->verts[i];

        set_output("res",std::move(res));
    }
};

ZENDEFNODE(AlignPrimitive,{
    {{"ref_shape"},{"aligned_shape"}},
    {"res"},
    {},
    {"Skinning"},
});


};