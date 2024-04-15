#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/PrimitiveObject.h>
// #include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>

#include <zeno/utils/vec.h>
#include <zeno/utils/eulerangle.h>

#include <glm/mat4x4.hpp>

namespace zeno {

struct CreateVolumeBox : zeno::INode {
    virtual void apply() override {

        auto pos = get_input2<zeno::vec3f>("pos");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");

        auto order = get_input2<std::string>("EulerRotationOrder:");
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

        auto measure = get_input2<std::string>("EulerAngleMeasure:");
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
        glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::mat4 transform(1.0f);
        
        if (has_input2<VDBGrid>("vdbGrid")) {

            auto grid = get_input2<VDBGrid>("vdbGrid");
		    auto box = grid->evalActiveVoxelBoundingBox();

		    glm::vec3 bmax = glm::vec3(box.max().x(), box.max().y(), box.max().z()) + 1.0f;
		    glm::vec3 bmin = glm::vec3(box.min().x(), box.min().y(), box.min().z());

            //printf("VDB BOX min= %f %f %f \n", bmin.x, bmin.y, bmin.z);
            //printf("VDB BOX max= %f %f %f \n", bmax.x, bmax.y, bmax.z);

            auto diff = bmax - bmin;
            auto center = bmin + diff / 2.0f;

            auto trans = glm::mat4(1.0f);

            trans = glm::translate(trans, center);
            trans = glm::scale(trans, diff);

            const auto world_matrix = [&]() -> auto {

                auto tmp = grid->getTransform().baseMap()->getAffineMap()->getMat4();
                glm::mat4 result;
                for (size_t i=0; i<16; ++i) {
                    auto ele = *(tmp[0]+i);
                    result[i/4][i%4] = ele;
                }
                return result;
            }();

            transform = world_matrix * trans;

        } else {

            transform = glm::translate(transform, glm::vec3(pos[0], pos[1], pos[2]));
            transform = transform * rotation;
            transform = glm::scale(transform, glm::vec3(scale[0], scale[1], scale[2]));
        }
        
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        float dummy[] = {-0.5f, 0.5f};

        for (int i=0; i<=1; ++i) {
            for (int j=0; j<=1; ++j) {
                for (int k=0; k<=1; ++k) {
                    auto p = glm::vec4(dummy[i], dummy[j], dummy[k], 1.0f);
                    p = transform * p; 
                    prim->verts.push_back(zeno::vec3f(p.x, p.y, p.z));
                }
            }
        }
        
        // enough to draw box wire frame
        prim->quads->push_back(zeno::vec4i(0, 1, 3, 2));
        prim->quads->push_back(zeno::vec4i(4, 5, 7, 6));
        prim->quads->push_back(zeno::vec4i(0, 1, 5, 4));
        prim->quads->push_back(zeno::vec4i(3, 2, 6, 7));

        primWireframe(prim.get(), true);
    
        auto transform_ptr = glm::value_ptr(transform);
            
            zeno::vec4f row0, row1, row2, row3;
            memcpy(row0.data(), transform_ptr, sizeof(float)*4);
            memcpy(row1.data(), transform_ptr+4, sizeof(float)*4);
            memcpy(row2.data(), transform_ptr+8, sizeof(float)*4);  
            memcpy(row3.data(), transform_ptr+12, sizeof(float)*4);

            prim->userData().set2("_transform_row0", row0);
            prim->userData().set2("_transform_row1", row1);
            prim->userData().set2("_transform_row2", row2);
            prim->userData().set2("_transform_row3", row3);

        prim->userData().set2("vbox", true);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateVolumeBox, {
    {
        {"vec3f", "pos", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"vdbGrid" },
    },
    {"prim"},
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {"create"}
});

} // namespace