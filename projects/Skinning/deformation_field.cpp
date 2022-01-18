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

// retrieve the deformation field based on the rest shape
struct RetrieveDeformationField : zeno::INode {
    virtual void apply() override {
        auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        auto def_shape = get_input<zeno::PrimitiveObject>("def_shape");

        size_t nm_elms = ref_shape->quads.size();

        auto def_field = std::make_shared<zeno::PrimitiveObject>();
        auto& sr0_field = def_field->add_attr<zeno::vec3f>("sr0");
        auto& sr1_field = def_field->add_attr<zeno::vec3f>("sr1");
        auto& sr2_field = def_field->add_attr<zeno::vec3f>("sr2");
        auto& t_field = def_field->add_attr<zeno::vec3f>("t");
        def_field->resize(nm_elms);

        for(size_t i = 0;i < nm_elms;++i){
            const auto& tet = ref_shape->quads[i];
            Eigen::Matrix4d ref_tet,def_tet;
            for(size_t j = 0;j < 4;++j){
                ref_tet.col(j) << ref_shape->verts[tet[j]][0],ref_shape->verts[tet[j]][1],ref_shape->verts[tet[j]][2],1.0;
                def_tet.col(j) << def_shape->verts[tet[j]][0],def_shape->verts[tet[j]][1],def_shape->verts[tet[j]][2],1.0;
            }

            auto A = def_tet * ref_tet.inverse();
            Eigen::Matrix3d R = A.block(0,0,3,3);

            auto q = Eigen::Quaternion<double>(R);
            auto b = Eigen::Vector3d(A(0,3),A(1,3),A(2,3));

            // qs_field[i] = zeno::vec4f(q.x(),q.y(),q.z(),q.w());
            sr0_field[i] = zeno::vec3f(A(0,0),A(0,1),A(0,2));
            sr1_field[i] = zeno::vec3f(A(1,0),A(1,1),A(1,2));
            sr2_field[i] = zeno::vec3f(A(2,0),A(2,1),A(2,2));
            t_field[i] = zeno::vec3f(A(0,3),A(1,3),A(2,3));
        }

        set_output("def_field",std::move(def_field));

    }
};

ZENDEFNODE(RetrieveDeformationField,{
    {{"ref_shape"},{"def_shape"}},
    {"def_field"},
    {},
    {"Skinning"},
});


// with relatively large rotation, directly interpolate on displacement field will be problematic, the skin apt to be shrinking
struct RetrieveDisplacementField : zeno::INode {
    virtual void apply() override {
        auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        auto def_shape = get_input<zeno::PrimitiveObject>("def_shape");

        auto disp_field = std::make_shared<zeno::PrimitiveObject>(*ref_shape);
        auto& disps = disp_field->add_attr<zeno::vec3f>("disp");
        disp_field->resize(ref_shape->size());

        if(ref_shape->size() != def_shape->size()){
            std::cerr << "THE SIZE OF TWO INPUT SHAPED DOES NOT ALIGN " << ref_shape->size() << "\t" << def_shape->size() << std::endl;
            throw std::runtime_error("THE SIZE OF TWO INPUT SHAPED DOES NOT ALIGN");
        }


        for(size_t i = 0;i < disp_field->size();++i){
            disps[i] = def_shape->verts[i] - ref_shape->verts[i];
        }

        set_output("disp_field",disp_field);      
    }
};

ZENDEFNODE(RetrieveDisplacementField,{
    {{"ref_shape"},{"def_shape"}},
    {"disp_field"},
    {},
    {"Skinning"},
});


// The two input primitive objects should be volumetric mesh, currently only quads is supported
// Using linear blending for interpolating the nodal affine field from neighbored elements
struct RetrieveAffineField : zeno::INode {
    virtual void apply() override {
        auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        auto def_shape = get_input<zeno::PrimitiveObject>("def_shape");

        assert(ref_shape->quads.size() == def_shape.quads.size());
        assert(ref_shape->size() == def_shape->size());

        size_t nm_quads = ref_shape->quads.size();

        std::vector<Eigen::Matrix4d> nodal_affine;
        nodal_affine.resize(ref_shape->size(),Eigen::Matrix4d::Zero());
        std::vector<double> nodal_weight_sum;
        nodal_weight_sum.resize(ref_shape->size(),0);

        for(size_t i = 0;i < nm_quads;++i){
            const auto& quad = ref_shape->quads[i];
            Eigen::Matrix4d ref_pos;
            Eigen::Matrix4d def_pos;

            std::vector<Eigen::Vector3d> elm_verts(4);

            for(size_t j = 0;j < 4;++j){
                size_t idx = quad[j];
                elm_verts[j] = Eigen::Vector3d(ref_shape->verts[idx][0],ref_shape->verts[idx][1],ref_shape->verts[idx][2]);
                ref_pos.col(j) << elm_verts[j],1.0;
                def_pos.col(j) << def_shape->verts[idx][0],def_shape->verts[idx][1],def_shape->verts[idx][2],1.0;
            }

            Eigen::Matrix4d affine = def_pos * ref_pos.inverse();
            // using the distance from the point to the opposite triangle as the weight
            for(size_t j = 0;j < 4;++j){
                double w = Height(elm_verts[j],elm_verts[(j+1)%4],elm_verts[(j+2)%4],elm_verts[(j+3)%4]);
                nodal_affine[quad[j]] += w * affine;
                nodal_weight_sum[quad[j]] += w;
            }
        }

        for(size_t i = 0;i < nodal_affine.size();++i)
            nodal_affine[i] /= nodal_weight_sum[i];

        auto affine_field = std::make_shared<zeno::PrimitiveObject>(*ref_shape);
        auto& affine0 = affine_field->add_attr<zeno::vec3f>("A0");
        auto& affine1 = affine_field->add_attr<zeno::vec3f>("A1");
        auto& affine2 = affine_field->add_attr<zeno::vec3f>("A2");
        auto& affine3 = affine_field->add_attr<zeno::vec3f>("A3");
        affine_field->resize(ref_shape->size());

        for(size_t i = 0;i < affine_field->size();++i){
            affine0[i] = zeno::vec3f(nodal_affine[i](0,0),nodal_affine[i](0,1),nodal_affine[i](0,2));
            affine1[i] = zeno::vec3f(nodal_affine[i](1,0),nodal_affine[i](1,1),nodal_affine[i](1,2));
            affine2[i] = zeno::vec3f(nodal_affine[i](2,0),nodal_affine[i](2,1),nodal_affine[i](2,2));
            affine3[i] = zeno::vec3f(nodal_affine[i](0,3),nodal_affine[i](1,3),nodal_affine[i](2,3));
        }

        // std::cout << "OUTPUT_AFFINE_FIELD " << std::endl;
        // for(size_t i= 0;i < affine_field->size();++i){
        //     std::cout << "ID : " <<  i << std::endl;
        //     std::cout << affine0[i][0] << "\t" << affine0[i][1] << "\t" << affine0[i][2] << std::endl;
        //     std::cout << affine1[i][0] << "\t" << affine1[i][1] << "\t" << affine1[i][2] << std::endl;
        //     std::cout << affine2[i][0] << "\t" << affine2[i][1] << "\t" << affine2[i][2] << std::endl;
        //     std::cout << affine3[i][0] << "\t" << affine3[i][1] << "\t" << affine3[i][2] << std::endl;
        // }

        set_output("affine_field",affine_field); 
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
    {{"ref_shape"},{"def_shape"}},
    {"affine_field"},
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