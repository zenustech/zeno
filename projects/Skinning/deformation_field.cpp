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


struct RetrieveAffineField : zeno::INode {
    virtual void apply() override {
        auto ref_shape = get_input<zeno::PrimitiveObject>("ref_shape");
        auto def_shape = get_input<zeno::PrimitiveObject>("def_shape");


    }
};

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