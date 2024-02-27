#include "zeno/types/UserData.h"
#include "zeno/funcs/ObjectGeometryInfo.h"
#include "zeno/types/ListObject.h"
#include "zeno/utils/log.h"
#include "zeno/funcs/PrimitiveUtils.h"
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/utils/eulerangle.h>
#include <zeno/utils/string.h>
#include <iostream>

namespace zeno {

static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
    auto vector4 = matrix * glm::vec4(vector, 1.0f);
    return glm::vec3(vector4) / vector4.w;
}
static glm::vec3 mapplynrm(glm::mat4 const &matrix, glm::vec3 const &vector) {
    glm::mat3 normMatrix(matrix);
    normMatrix = glm::transpose(glm::inverse(normMatrix));
    auto vector3 = normMatrix * vector;
    return glm::normalize(vector3);
}

ZENO_API void transformPrimObj(PrimitiveObject *prim, glm::mat4 parent_matrix) {
    auto matrix = parent_matrix * prim->transformMat;
    prim->transformMat = glm::mat4(1);
    if (matrix != glm::mat4(1)) {
        if (prim->has_attr("pos")) {
            auto &pos = prim->attr<zeno::vec3f>("pos");
            prim->verts.add_attr<zeno::vec3f>("_origin_pos") = pos;
            #pragma omp parallel for
            for (int i = 0; i < pos.size(); i++) {
                auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                p = mapplypos(matrix, p);
                pos[i] = zeno::other_to_vec<3>(p);
            }
        }

        if (prim->has_attr("nrm")) {
            auto &nrm = prim->attr<zeno::vec3f>("nrm");
            prim->verts.add_attr<zeno::vec3f>("_origin_nrm") = nrm;
            #pragma omp parallel for
            for (int i = 0; i < nrm.size(); i++) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                n = mapplynrm(matrix, n);
                nrm[i] = zeno::other_to_vec<3>(n);
            }
        }
    }

}
namespace {

struct SetMatrix : zeno::INode{//ZHXX: use Assign instead!
    virtual void apply() override {
        auto &dst = std::get<glm::mat4>(get_input<zeno::MatrixObject>("dst")->m);
        auto &src = std::get<glm::mat4>(get_input<zeno::MatrixObject>("src")->m);
        dst = src;
    }
};
ZENDEFNODE(SetMatrix, {
    {
    {"dst" },
    {"src" },
    },
    {},
    {},
    {"math"},
});

struct MakeLocalSys : zeno::INode{
    virtual void apply() override {
        zeno::vec3f front = {1,0,0};
        zeno::vec3f up = {0,1,0};
        zeno::vec3f right = {0,0,1};
        if (has_input("front"))
            front = get_input<zeno::NumericObject>("front")->get<zeno::vec3f>();
        if (has_input("up"))
            up = get_input<zeno::NumericObject>("up")->get<zeno::vec3f>();
        if (has_input("right"))
            right = get_input<zeno::NumericObject>("right")->get<zeno::vec3f>();

        auto oMat = std::make_shared<MatrixObject>();
        oMat->m = glm::mat4(glm::mat3(front[0], up[0], right[0],
                            front[1], up[1], right[1],
                            front[2], up[2], right[2]));
        set_output("LocalSys", oMat);                    
    }
};
ZENDEFNODE(MakeLocalSys, {
    {
    {"vec3f", "front", "1,0,0"},
    {"vec3f", "up", "0,1,0"},
    {"vec3f", "right", "0,0,1"},
    },
    {{"LocalSys"}},
    {},
    {"math"},
});

struct TransformPrimitive : zeno::INode {//zhxx happy node
    static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
        auto vector4 = matrix * glm::vec4(vector, 1.0f);
        return glm::vec3(vector4) / vector4.w;
    }


    static glm::vec3 mapplynrm(glm::mat4 const &matrix, glm::vec3 const &vector) {
        glm::mat3 normMatrix(matrix);
        normMatrix = glm::transpose(glm::inverse(normMatrix));
        auto vector3 = normMatrix * vector;
        return glm::normalize(vector3);
    }

    virtual void apply() override {
        zeno::vec3f translate = {0,0,0};
        zeno::vec4f rotation = {0,0,0,1};
        zeno::vec3f eulerXYZ = {0,0,0};
        zeno::vec3f scaling = {1,1,1};
        zeno::vec3f shear = {0,0,0};
        zeno::vec3f offset = {0,0,0};
        glm::mat4 pre_mat = glm::mat4(1.0);
        glm::mat4 pre_apply = glm::mat4(1.0);
        glm::mat4 local = glm::mat4(1.0);
        if (has_input("Matrix"))
            pre_mat = std::get<glm::mat4>(get_input<zeno::MatrixObject>("Matrix")->m);
        if (has_input("translation"))
            translate = get_input<zeno::NumericObject>("translation")->get<zeno::vec3f>();
        if (has_input("eulerXYZ"))
            eulerXYZ = get_input<zeno::NumericObject>("eulerXYZ")->get<zeno::vec3f>();
        if (has_input("quatRotation"))
            rotation = get_input<zeno::NumericObject>("quatRotation")->get<zeno::vec4f>();
        if (has_input("scaling"))
            scaling = get_input<zeno::NumericObject>("scaling")->get<zeno::vec3f>();
        if (has_input("shear"))
            shear = get_input<zeno::NumericObject>("shear")->get<zeno::vec3f>();
        if (has_input("offset"))
            offset = get_input<zeno::NumericObject>("offset")->get<zeno::vec3f>();
        if (has_input("local"))
           local = std::get<glm::mat4>(get_input<zeno::MatrixObject>("local")->m);
        if (has_input("preTransform"))
            pre_apply = std::get<glm::mat4>(get_input<zeno::MatrixObject>("preTransform")->m);

        glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(eulerXYZ[0], eulerXYZ[1], eulerXYZ[2]);
            glm::mat4 matRotate = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
        glm::mat4 matQuat  = glm::toMat4(myQuat);
        glm::mat4 matScal  = glm::scale( glm::vec3(scaling[0], scaling[1], scaling[2] ));
        glm::mat4 matShearX = glm::transpose(glm::mat4(
            1, shear[0], 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1));
        glm::mat4 matShearY = glm::transpose(glm::mat4(
            1, 0, shear[1], 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1));
        glm::mat4 matShearZ = glm::transpose(glm::mat4(
            1, 0, 0, 0,
            0, 1, shear[2], 0,
            0, 0, 1, 0,
            0, 0, 0, 1));

        auto matrix = pre_mat*local*matTrans*matRotate*matQuat*matScal*matShearZ*matShearY*matShearX*glm::translate(glm::vec3(offset[0], offset[1], offset[2]))*glm::inverse(local)*pre_apply;

        auto prim = get_input<PrimitiveObject>("prim");
        auto outprim = std::make_unique<PrimitiveObject>(*prim);

        if (prim->has_attr("pos")) {
            auto &pos = outprim->attr<zeno::vec3f>("pos");
            #pragma omp parallel for
            for (int i = 0; i < pos.size(); i++) {
                auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                p = mapplypos(matrix, p);
                pos[i] = zeno::other_to_vec<3>(p);
            }
        }

        if (prim->has_attr("nrm")) {
            auto &nrm = outprim->attr<zeno::vec3f>("nrm");
            #pragma omp parallel for
            for (int i = 0; i < nrm.size(); i++) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                n = mapplynrm(matrix, n);
                nrm[i] = zeno::other_to_vec<3>(n);
            }
        }

        auto& user_data = outprim->userData();
        user_data.setLiterial("_translate", translate);
        vec4f rotate = {myQuat.x, myQuat.y, myQuat.z, myQuat.w};
        user_data.setLiterial("_rotate", rotate);
        user_data.setLiterial("_scale", scaling);
        //auto oMat = std::make_shared<MatrixObject>();
        //oMat->m = matrix;
        set_output("outPrim", std::move(outprim));
    }
};

ZENDEFNODE(TransformPrimitive, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "translation", "0,0,0"},
    {"vec3f", "eulerXYZ", "0,0,0"},
    {"vec4f", "quatRotation", "0,0,0,1"},
    {"vec3f", "scaling", "1,1,1"},
    {"vec3f", "shear", "0,0,0"},
    {"Matrix"},
    {"preTransform"},
    {"local"},
    },
    {
    {"PrimitiveObject", "outPrim"}
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "ZYX"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", EulerAngle::MeasureDefaultString()}
    },
    {"deprecated"},
});

static void printMat4(const glm::mat4& matrix) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
// euler rot order: roll-pitch-yaw
// euler rot unit use degrees
struct PrimitiveTransform : zeno::INode {


    static std::optional<std::shared_ptr<IObject>> get_from_list(std::string path, std::shared_ptr<IObject> iObject, std::map<std::string, std::string>& listitemidxs) {
        if (path.empty() || path == "/") {
            return iObject;
        }
        auto cur_root = iObject;
        if (auto rootlst = std::dynamic_pointer_cast<ListObject>(cur_root)) {
            path = listitemidxs[path];
        }

        auto idxs = split_str(path, '/');
        std::vector<int> idxs_int;
        for (const auto &idx: idxs) {
            if (idx.empty()) {
                continue;
            }
            auto i = std::stoi(idx);
            if (auto list = std::dynamic_pointer_cast<ListObject>(cur_root)) {
                if (i >= list->arr.size()) {
                    zeno::log_warn("out of range");
                    return std::nullopt;
                }
                cur_root = list->arr[i];
            }
            else {
                return cur_root;
            }
        }
        return cur_root;
    }

    static void transformObj(
            std::shared_ptr<IObject> iObject
            , glm::mat4 matrix
            , std::string pivotType
            , vec3f pivotPos
            , vec3f translate
            , vec4f rotation
            , vec3f scaling
    ) {
        if (auto prim = std::dynamic_pointer_cast<PrimitiveObject>(iObject)) {

            zeno::vec3f _pivot = {};
            if (pivotType == "bboxCenter") {
                zeno::vec3f _min;
                zeno::vec3f _max;
                std::tie(_min, _max) = primBoundingBox(prim.get());
                _pivot = (_min + _max) / 2;
            }
            else if (pivotType == "custom") {
                _pivot = pivotPos;
            }
            auto pivot_to_local = glm::translate(glm::vec3(-_pivot[0], -_pivot[1], -_pivot[2]));
            auto pivot_to_world = glm::translate(glm::vec3(_pivot[0], _pivot[1], _pivot[2]));
            prim->transformMat = pivot_to_world * matrix * pivot_to_local;
            printMat4(prim->transformMat);

            auto& user_data = prim->userData();
            user_data.setLiterial("_translate", translate);
            user_data.setLiterial("_rotate", rotation);
            user_data.setLiterial("_scale", scaling);
            user_data.set2("_pivot", _pivot);
            user_data.del("_bboxMin");
            user_data.del("_bboxMax");
        }
        else if (auto list = std::dynamic_pointer_cast<ListObject>(iObject)) {
            for (auto &item : list->arr) {
                transformObj(item, matrix, pivotType, translate, pivotPos, rotation, scaling);
            }
        }
    }
    virtual void apply() override {
        zeno::vec3f translate = {0,0,0};
        zeno::vec4f rotation = {0,0,0,1};
        zeno::vec3f eulerXYZ = {0,0,0};
        zeno::vec3f scaling = {1,1,1};
        zeno::vec3f shear = {0,0,0};
        zeno::vec3f offset = {0,0,0};
        glm::mat4 pre_mat = glm::mat4(1.0);
        glm::mat4 pre_apply = glm::mat4(1.0);
        glm::mat4 local = glm::mat4(1.0);
        if (has_input("Matrix"))
            pre_mat = std::get<glm::mat4>(get_input<zeno::MatrixObject>("Matrix")->m);
        if (has_input("translation"))
            translate = get_input<zeno::NumericObject>("translation")->get<zeno::vec3f>();
        if (has_input("eulerXYZ"))
            eulerXYZ = get_input<zeno::NumericObject>("eulerXYZ")->get<zeno::vec3f>();
        if (has_input("quatRotation"))
            rotation = get_input<zeno::NumericObject>("quatRotation")->get<zeno::vec4f>();
        if (has_input("scaling"))
            scaling = get_input<zeno::NumericObject>("scaling")->get<zeno::vec3f>();
        if (has_input("shear"))
            shear = get_input<zeno::NumericObject>("shear")->get<zeno::vec3f>();
        if (has_input("offset"))
            offset = get_input<zeno::NumericObject>("offset")->get<zeno::vec3f>();
        if (has_input("local"))
            local = std::get<glm::mat4>(get_input<zeno::MatrixObject>("local")->m);
        if (has_input("preTransform"))
            pre_apply = std::get<glm::mat4>(get_input<zeno::MatrixObject>("preTransform")->m);

        glm::mat4 matTrans = glm::translate(glm::vec3(translate[0], translate[1], translate[2]));

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(eulerXYZ[0], eulerXYZ[1], eulerXYZ[2]);
            glm::mat4 matRotate = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::quat myQuat(rotation[3], rotation[0], rotation[1], rotation[2]);
        glm::mat4 matQuat  = glm::toMat4(myQuat);
        glm::mat4 matScal  = glm::scale( glm::vec3(scaling[0], scaling[1], scaling[2] ));
        glm::mat4 matShearX = glm::transpose(glm::mat4(
                1, shear[0], 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1));
        glm::mat4 matShearY = glm::transpose(glm::mat4(
                1, 0, shear[1], 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1));
        glm::mat4 matShearZ = glm::transpose(glm::mat4(
                1, 0, 0, 0,
                0, 1, shear[2], 0,
                0, 0, 1, 0,
                0, 0, 0, 1));

        auto matrix = pre_mat*local*matTrans*matRotate*matQuat*matScal*matShearZ*matShearY*matShearX*glm::translate(glm::vec3(offset[0], offset[1], offset[2]))*glm::inverse(local)*pre_apply;

        auto iObject = get_input2<IObject>("prim");
        auto path = get_input2<std::string>("path");

        std::string pivotType = get_input2<std::string>("pivot");

        auto pivotPos = get_input2<vec3f>("pivotPos");

        if (std::dynamic_pointer_cast<PrimitiveObject>(iObject)) {
            iObject = iObject->clone();
            transformObj(iObject, matrix, pivotType, pivotPos, translate, rotation, scaling);
            printMat4(std::dynamic_pointer_cast<PrimitiveObject>(iObject)->transformMat);
        }
        else {
            if (path != "")
            {
                std::map<std::string, std::string> listitemidxsMap;
                std::string listitemidxs = iObject->userData().get2<std::string>("list-item-idxs", "");
                listitemidxs.erase(listitemidxs.length() - 1);
                for (const auto& idx : split_str(listitemidxs, ';'))
                    listitemidxsMap.insert(std::make_pair(idx.substr(0, idx.find_first_of(":")), idx.substr(idx.find_first_of(":") + 1)));

                path.erase(path.length() - 1);
                auto idxs = split_str(path, ';');
                for (const auto& idx : idxs) {
                    auto select = get_from_list(idx, iObject, listitemidxsMap);
                    if (select.has_value()) {
                        transformObj(select.value(), matrix, pivotType, pivotPos, translate, rotation, scaling);
                    }
                }
            }
        }

        set_output("outPrim", std::move(iObject));
    }
};

ZENDEFNODE(PrimitiveTransform, {
    {
        {"PrimitiveObject", "prim"},
        {"string", "path"},
        {"vec3f", "translation", "0,0,0"},
        {"vec3f", "eulerXYZ", "0,0,0"},
        {"vec4f", "quatRotation", "0,0,0,1"},
        {"vec3f", "scaling", "1,1,1"},
        {"vec3f", "shear", "0,0,0"},
        {"enum world bboxCenter custom", "pivot", "bboxCenter"},
        {"vec3f", "pivotPos", "0,0,0"},
        {"Matrix"},
        {"preTransform"},
        {"local"},
    },
    {
        {"PrimitiveObject", "outPrim"}
    },
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", EulerAngle::RotationOrderDefaultString()},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {"primitive"},
});

struct PrimitiveCollapse : zeno::INode {
    static void transformObj(std::shared_ptr<IObject> iObject, glm::mat4 parent_matrix = glm::mat4(1)) {
        if (auto prim = std::dynamic_pointer_cast<PrimitiveObject>(iObject)) {
            transformPrimObj(prim.get());
        }
        else {
            auto list = std::dynamic_pointer_cast<ListObject>(iObject);
            for (auto &item : list->arr) {
                transformObj(item, list->transformMat);
            }
        }
    }
    virtual void apply() override {
        auto iObject = get_input2<IObject>("in");
        transformObj(iObject);
        set_output("outPrim", std::move(iObject));
    }
};

ZENDEFNODE(PrimitiveCollapse, {
    {
        "in"
    },
    {
        "out"
    },
    {
    },
    {"primitive"},
});

}
}
