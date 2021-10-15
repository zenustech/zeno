
#include <matrix_helper.hpp>

namespace zeno{
    struct TransformMatrix : zeno::IObject {
        TransformMatrix() = default;
        Mat4x4d Mat;
    };

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


