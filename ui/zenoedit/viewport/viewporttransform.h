//
// Created by ryan on 22-8-8.
//

#ifndef __VIEWPORT_TRANSFORM_H__
#define __VIEWPORT_TRANSFORM_H__
#include "zenovis.h"

#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>

#include <QtWidgets>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <unordered_map>

namespace zeno {

class FakeTransformer {
  public:
    FakeTransformer()
    : transform_matrix(1.0f)
    , scale_matrix(1.0f)
    , rotate_matrix(1.0f)
    , translate_matrix(1.0f)
    , step_transform_matrix(1.0f)
    , objects_center(0.0f)
    , status(false)
    , rotate_eular_vec(0.0f)
    {

    }

    FakeTransformer(const std::unordered_set<std::string>& names)
    : transform_matrix(1.0f)
    , scale_matrix(1.0f)
    , rotate_matrix(1.0f)
    , translate_matrix(1.0f)
    , step_transform_matrix(1.0f)
    , objects_center(0.0f)
    , status(true)
    , rotate_eular_vec(0.0f)
    {
        addObject(names);
    }

    void addObject(const std::string& name) {
        auto nm = name.substr(0, name.find_first_of(':'));
        // printf(nm.c_str());
        // printf("\n");
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        auto object = dynamic_cast<PrimitiveObject*>(scene->objectsMan->get(name).value());
        objects_center *= objects.size();
        auto user_data = object->userData();
        zeno::vec3f bmin, bmax;
        if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
            bmin = user_data.getLiterial<zeno::vec3f>("_bboxMin");
            bmax = user_data.getLiterial<zeno::vec3f>("_bboxMax");
        } else {
            std::tie(bmin, bmax) = zeno::primBoundingBox(object);
            user_data.setLiterial("_bboxMin", bmin);
            user_data.setLiterial("_bboxMax", bmax);
        }
        auto m = zeno::vec_to_other<glm::vec3>(bmax);
        auto n = zeno::vec_to_other<glm::vec3>(bmin);
        objects_center += (m + n) / 2.0f;
        objects[name] = object;
        objects_center /= objects.size();
    }

    void addObject(const std::unordered_set<std::string>& names) {
        for (const auto& name : names) {
            addObject(name);
        }
    }

    void removeObject(const std::string& name) {
        auto p = objects.find(name);
        if (p == objects.end())
            return ;
        auto object = p->second;
        objects_center *= objects.size();
        auto user_data = object->userData();
        zeno::vec3f bmin, bmax;
        if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
            bmin = user_data.getLiterial<zeno::vec3f>("_bboxMin");
            bmax = user_data.getLiterial<zeno::vec3f>("_bboxMax");
        } else {
            std::tie(bmin, bmax) = zeno::primBoundingBox(object);
            user_data.setLiterial("_bboxMin", bmin);
            user_data.setLiterial("_bboxMax", bmax);
        }
        auto m = zeno::vec_to_other<glm::vec3>(bmax);
        auto n = zeno::vec_to_other<glm::vec3>(bmin);
        objects_center -= (m + n) / 2.0f;
        objects.erase(p);
        objects_center /= objects.size();
    }

    void removeObject(const std::unordered_set<std::string>& names) {
        for (const auto& name : names) {
            removeObject(name);
        }
    }

    void translate(QVector3D start, QVector3D end, QVector3D axis) {
        auto diff = end - start;
        glm::vec3 diff_vec3 = QVec3ToGLMVec3(diff);
        diff_vec3 *= QVec3ToGLMVec3(axis);
        step_transform_matrix = glm::translate(diff_vec3);
        translate_matrix = step_transform_matrix * translate_matrix;
        transform_matrix = step_transform_matrix * transform_matrix;
        doTransform();
    }

    void print_mat4(glm::mat4 mat) const {
        printf("\n");
        for (int i=0; i<4; i++) {
            for (int j=0; j<4; j++)
                printf("%.2lf ", transform_matrix[i][j]);
            printf("\n");
        }
    }

    void scale(QVector3D cur_vec, vec3i axis, float scale_size) {
        glm::vec3 cur_vec3 = QVec3ToGLMVec3(cur_vec);
        glm::vec3 scale(1.0f);
        for (int i=0; i<3; i++)
            if (axis[i] == 1) scale[i] = scale_size * 3;
        auto inv_last_scale = glm::inverse(scale_matrix);
        scale_matrix = glm::scale(scale);
        auto translate_to_origin = glm::translate(-objects_center);
        auto translate_to_center = glm::translate(objects_center);
        step_transform_matrix = translate_to_center * scale_matrix * inv_last_scale * translate_to_origin;
        transform_matrix = step_transform_matrix * transform_matrix;
        doTransform();
    }

    void rotate(QVector3D start_vec, QVector3D end_vec, QVector3D axis) {
        glm::vec3 start_vec3 = QVec3ToGLMVec3(start_vec);
        glm::vec3 end_vec3 = QVec3ToGLMVec3(end_vec);
        glm::vec3 axis_vec3 = QVec3ToGLMVec3(axis);
        start_vec3 = glm::normalize(start_vec3);
        end_vec3 = glm::normalize(end_vec3);
        auto cross_vec3 = glm::cross(start_vec3, end_vec3);
        float direct = 1.0f;
        if (glm::dot(cross_vec3, axis_vec3) < 0)
            direct = -1.0f;
        float angle = acos(fmin(fmax(glm::dot(start_vec3, end_vec3), -1.0f), 1.0f));
        rotate_eular_vec += angle * direct * axis_vec3;
        step_transform_matrix = glm::rotate(angle * direct, axis_vec3);
        auto translate_to_origin = glm::translate(-objects_center);
        auto translate_to_center = glm::translate(objects_center);
        step_transform_matrix = translate_to_center * step_transform_matrix * translate_to_origin;
        rotate_matrix = step_transform_matrix * rotate_matrix;
        transform_matrix = step_transform_matrix * transform_matrix;
        doTransform();
    }

    glm::vec3 getTranslateVec() const {
        return {translate_matrix[3][0], translate_matrix[3][1], translate_matrix[3][2]};
    }

    glm::vec3 getScaleVec() const {
        return {scale_matrix[0][0], scale_matrix[1][1], scale_matrix[2][2]};
    }

    glm::vec3 getRotateEulerVec() const {
        return rotate_eular_vec;
    }

    void startTransform() {
        status = true;
        Zenovis::GetInstance().getSession()->set_interactive(true);
    }

    void endTransform() {
        status = false;
        translate_matrix = glm::mat4(1.0f);
        scale_matrix = glm::mat4(1.0f);
        rotate_eular_vec = glm::vec3(1.0f);
        Zenovis::GetInstance().getSession()->set_interactive(false);
    }

    bool isTransforming() const {
        return status;
    }

    glm::vec3 getCenter() {
        return objects_center;
    }

    void clear() {
        objects.clear();
        objects_center = {0, 0, 0};
        transform_matrix = glm::mat4(1.0f);
        scale_matrix = glm::mat4(1.0f);
        rotate_matrix = glm::mat4(1.0f);
        translate_matrix = glm::mat4(1.0f);
        step_transform_matrix = glm::mat4(1.0f);
        objects_center = {0, 0, 0};
        rotate_eular_vec = {0, 0, 0};
    }

  private:
    static glm::vec3 QVec3ToGLMVec3(QVector3D QVec3) {
        return {QVec3.x(), QVec3.y(), QVec3.z()};
    }

    void doTransform() {
        for (auto &[obj_name, obj] : objects) {
            if (obj->has_attr("pos")) {
                // transform pos
                auto &pos = obj->attr<zeno::vec3f>("pos");
                for (auto &po : pos) {
                    auto p = zeno::vec_to_other<glm::vec3>(po);
                    auto t = step_transform_matrix * glm::vec4(p, 1.0f);
                    auto pt = glm::vec3(t) / t.w;
                    po = zeno::other_to_vec<3>(pt);
                }
            }
            if (obj->has_attr("nrm")) {
                // transform nrm
                auto &nrm = obj->attr<zeno::vec3f>("nrm");
                for (auto &i : nrm) {
                    auto n = zeno::vec_to_other<glm::vec3>(i);
                    glm::mat3 norm_matrix(step_transform_matrix);
                    norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                    auto t = glm::normalize(norm_matrix * n);
                    i = zeno::other_to_vec<3>(n);
                }
            }
            auto &user_data = obj->userData();
            if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
                vec3f bmin, bmax;
                std::tie(bmin, bmax) = primBoundingBox(obj);
                user_data.setLiterial("_bboxMin", bmin);
                user_data.setLiterial("_bboxMax", bmax);
            }
        }
        // update center
        glm::vec4 center(objects_center, 1);
        center = step_transform_matrix * center;
        center /= center[3];
        objects_center = center;
    }

  private:
    std::unordered_map<std::string, PrimitiveObject*> objects;

    glm::mat4 translate_matrix;
    glm::mat4 scale_matrix;
    glm::mat4 rotate_matrix;
    glm::mat4 transform_matrix;
    glm::mat4 step_transform_matrix;

    glm::vec3 rotate_eular_vec;

    glm::vec3 objects_center;

    bool status;
};

//void getTranslateTransformMatrix(glm::mat4 &matrix, QVector3D start, QVector3D end, QVector3D axis);
//void getTranslateTransformMatrix(glm::mat4 &matrix, QVector3D start, QVector3D end, QVector3D axis) {
//    auto diff = end - start;
//    glm::vec3 diff_vec3 = QVec3ToGLMVec3(diff);
//    diff_vec3 *= QVec3ToGLMVec3(axis);
//    matrix = glm::translate(diff_vec3);
//}
//
////void print_vec3(std::string name, glm::vec3 vec) {
////    printf("%s: %.2lf, %.2lf, %.2lf // ", name.c_str(), vec[0], vec[1], vec[2]);
////}
//
//void getRotateTransformMatrix(glm::mat4 &matrix, QVector3D start_vec, QVector3D end_vec, QVector3D axis);
//void getRotateTransformMatrix(glm::mat4 &matrix, QVector3D start_vec, QVector3D end_vec, QVector3D axis) {
//    glm::vec3 start_vec3 = QVec3ToGLMVec3(start_vec);
//    glm::vec3 end_vec3 = QVec3ToGLMVec3(end_vec);
//    glm::vec3 axis_vec3 = QVec3ToGLMVec3(axis);
//    start_vec3 = glm::normalize(start_vec3);
//    end_vec3 = glm::normalize(end_vec3);
//    auto cross_vec3 = glm::cross(start_vec3, end_vec3);
//    float direct = 1.0f;
//    if (glm::dot(cross_vec3, axis_vec3) < 0)
//        direct = -1.0f;
//    float angle = acos(fmin(fmax(glm::dot(start_vec3, end_vec3), -1.0f), 1.0f));
//    matrix = glm::rotate(angle * direct, axis_vec3);
//}
//
//void getScaleTransformMatrix(glm::mat4 &matrix, QVector3D cur_vec, vec3i axis, float scale_size);
//void getScaleTransformMatrix(glm::mat4 &matrix, QVector3D cur_vec, vec3i axis, float scale_size) {
//    glm::vec3 cur_vec3 = QVec3ToGLMVec3(cur_vec);
//    glm::vec3 scale(1.0f);
//    for (int i=0; i<3; i++)
//        if (axis[i] == 1) scale[i] = scale_size * 3;
//    matrix = glm::scale(scale);
//}
//
//void doFakeTransformPrimitive(zeno::PrimitiveObject *obj, glm::mat4 transform_matrix);
//void doFakeTransformPrimitive(zeno::PrimitiveObject *obj, glm::mat4 const transform_matrix) {
//
//    if (obj->has_attr("pos")) {
//        // transform pos
//        auto &pos = obj->attr<zeno::vec3f>("pos");
//        for (auto &po : pos) {
//            auto p = zeno::vec_to_other<glm::vec3>(po);
//            auto t = transform_matrix * glm::vec4(p, 1.0f);
//            auto pt = glm::vec3(t) / t.w;
//            po = zeno::other_to_vec<3>(pt);
//        }
//    }
//    if (obj->has_attr("nrm")) {
//        // transform nrm
//        auto &nrm = obj->attr<zeno::vec3f>("nrm");
//        for (auto &i : nrm) {
//            auto n = zeno::vec_to_other<glm::vec3>(i);
//            glm::mat3 norm_matrix(transform_matrix);
//            norm_matrix = glm::transpose(glm::inverse(norm_matrix));
//            auto t = glm::normalize(norm_matrix * n);
//            i = zeno::other_to_vec<3>(n);
//        }
//    }
//    auto &user_data = obj->userData();
//    if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
//        vec3f bmin, bmax;
//        std::tie(bmin, bmax) = primBoundingBox(obj);
//        user_data.setLiterial("_bboxMin", bmin);
//        user_data.setLiterial("_bboxMax", bmax);
//    }
//}

}

#endif //__VIEWPORT_TRANSFORM_H__
