
#include <matrix_helper.hpp>
#include <cmath>

namespace zeno{
    struct TransformMatrix : zeno::IObject {
        TransformMatrix() = default;
        Mat4x4d Mat;
    };

    struct MakeTransformMatrix : zeno::INode {
        virtual void apply() override {
            auto euler_xyz = get_input<zeno::NumericObject>("eulerAngle")->get<zeno::vec3f>();
            auto t = get_input<zeno::NumericObject>("translate")->get<zeno::vec3f>();

            double s1 = sin(euler_xyz[0]);
            double c1 = cos(euler_xyz[0]);
            double s2 = sin(euler_xyz[1]);
            double c2 = cos(euler_xyz[1]);
            double s3 = sin(euler_xyz[2]);
            double c3 = cos(euler_xyz[2]);

            auto TM = std::make_shared<TransformMatrix>();
            TM->Mat <<  c2*c3,          -s2,        c2*s3,          t[0],
                        s1*s3+c1*c3*s2, c1*c2,      c1*s2*s3-c3*s1, t[1],
                        c3*s1*s2-c1*s3, c2*s1,      c1*c3+s1*s2*s3, t[2],
                        0,              0,          0,              1.0;
            set_output("TM",std::move(TM));
        }
    };

    ZENDEFNODE(MakeTransformMatrix, {
        {"eulerAngle","translate"},
        {"TM"},
        {},
        {"FEM"},
    });

    struct MakeIdentityMatrix : zeno::INode {
        virtual void apply() override {
            auto ret = std::make_shared<TransformMatrix>();
            ret->Mat = Mat4x4d::Identity();

            set_output("ret",std::move(ret));
        }
    };

    ZENDEFNODE(MakeIdentityMatrix, {
        {},
        {"ret"},
        {},
        {"FEM"},
    });

};


