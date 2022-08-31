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
#include <nodesys/nodesmgr.h>

#include <QtWidgets>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <unordered_map>

namespace zeno {

enum {
    TRANSLATE,
    ROTATE,
    SCALE,
    NONE
};

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
    , operation(NONE)
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
    , operation(NONE)
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
        operation = TRANSLATE;
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
        operation = SCALE;
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
        operation = ROTATE;
    }

    inline glm::vec3 getTranslateVec() const {
        return {translate_matrix[3][0], translate_matrix[3][1], translate_matrix[3][2]};
    }

    inline glm::vec3 getScaleVec() const {
        return {scale_matrix[0][0], scale_matrix[1][1], scale_matrix[2][2]};
    }

    inline glm::vec3 getRotateEulerVec() const {
        return rotate_eular_vec;
    }

    void createNewTransformNode(QString& node_id, IGraphsModel* pModel, QModelIndex& node_index, QModelIndex& subgraph_index, bool change_visibility=true) const {
        auto pos = node_index.data(ROLE_OBJPOS).toPointF();
        pos.setX(pos.x() + 10);
        auto new_node_id = NodesMgr::createNewNode(pModel, subgraph_index, "TransformPrimitive", pos);
        QString out_sock = "prim";
        if (node_id.contains("TransformPrimitive"))
            out_sock = "outPrim";
        EdgeInfo edge = {
            node_id,
            new_node_id,
            out_sock,
            "prim"
        };
        pModel->addLink(edge, subgraph_index, false);

        auto translate_vec3 = getTranslateVec();
        QVector<double> translate = {
            translate_vec3[0],
            translate_vec3[1],
            translate_vec3[2]
        };
        PARAM_UPDATE_INFO translate_info = {
            "translation",
            QVariant::fromValue(QVector<double>{0, 0, 0}),
            QVariant::fromValue(translate)
        };
        pModel->updateSocketDefl(new_node_id, translate_info, subgraph_index, true);

        auto scaling_vec3 = getScaleVec();
        QVector<double> scaling = {
            scaling_vec3[0],
            scaling_vec3[1],
            scaling_vec3[2]
        };
        PARAM_UPDATE_INFO scaling_info = {
            "scaling",
            QVariant::fromValue(QVector<double>{1, 1, 1}),
            QVariant::fromValue(scaling)
        };
        pModel->updateSocketDefl(new_node_id, scaling_info, subgraph_index, true);
        auto rotate_vec3 = getRotateEulerVec();
        QVector<double> rotate = {
            rotate_vec3[0],
            rotate_vec3[1],
            rotate_vec3[2]
        };
        PARAM_UPDATE_INFO rotate_info = {
            "eulerXYZ",
            QVariant::fromValue(QVector<double>{0, 0, 0}),
            QVariant::fromValue(rotate)
        };
        pModel->updateSocketDefl(new_node_id, rotate_info, subgraph_index, true);

        if (change_visibility) {
            // make node not visible
            int old_option = node_index.data(ROLE_OPTIONS).toInt();
            int new_option = old_option;
            new_option ^= OPT_VIEW;
            STATUS_UPDATE_INFO status_info = {old_option, new_option, ROLE_OPTIONS};
            pModel->updateNodeStatus(node_id, status_info, subgraph_index, true);

            // make new node visible
            old_option = 0;
            new_option = 0 | OPT_VIEW;
            status_info.oldValue = old_option;
            status_info.newValue = new_option;
            pModel->updateNodeStatus(new_node_id, status_info, subgraph_index, true);
        }
    }

    QVariant linkedToVisibleTransformNode(QModelIndex& node_index, IGraphsModel* pModel) const {
        auto output_sockets = node_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        auto linked_edges = output_sockets["prim"].linkIndice;
        for (const auto& linked_edge : linked_edges) {
            auto node_id = linked_edge.data(ROLE_INNODE).toString();
            if (node_id.contains("TransformPrimitive")) {
                auto search_result = pModel->search(node_id, SEARCH_NODEID);
                auto linked_node_index = search_result[0].targetIdx;
                auto option = linked_node_index.data(ROLE_OPTIONS).toInt();
                if (option & OPT_VIEW) {
                    return QVariant::fromValue(linked_node_index);
                }
            }
        }
        return {};
    }

    void syncToTransformNode(QString& node_id, IGraphsModel* pModel, QModelIndex& node_index, QModelIndex& subgraph_index) const {
        auto inputs = node_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        auto translate_old = inputs["translation"].info.defaultValue.value<UI_VECTYPE>();
        QVector<double> translate_new(translate_old);
        auto translate_offset = getTranslateVec();
        for (int i=0; i<3; i++) {
            translate_new[i] += translate_offset[i];
        }
        PARAM_UPDATE_INFO translate_info = {
            "translation",
            QVariant::fromValue(translate_old),
            QVariant::fromValue(translate_new)
        };
        pModel->updateSocketDefl(node_id, translate_info, subgraph_index, true);
        // update scaling
        auto scaling_old = inputs["scaling"].info.defaultValue.value<UI_VECTYPE>();
        QVector<double> scaling_new(scaling_old);
        auto scaling_offset = getScaleVec();
        for (int i=0; i<3; i++) {
            scaling_new[i] *= scaling_offset[i];
        }
        PARAM_UPDATE_INFO scaling_info = {
            "scaling",
            QVariant::fromValue(scaling_old),
            QVariant::fromValue(scaling_new)
        };
        pModel->updateSocketDefl(node_id, scaling_info, subgraph_index, true);
        // update rotate
        auto rotate_old = inputs["eulerXYZ"].info.defaultValue.value<UI_VECTYPE>();
        QVector<double> rotate_new(rotate_old);
        auto rotate_offset = getRotateEulerVec();
        for (int i=0; i<3; i++) {
            rotate_new[i] += rotate_offset[i];
        }
        PARAM_UPDATE_INFO rotate_info = {
            "eulerXYZ",
            QVariant::fromValue(rotate_old),
            QVariant::fromValue(rotate_new)
        };
        pModel->updateSocketDefl(node_id, rotate_info, subgraph_index, true);
    }

    void startTransform() {
        status = true;
        Zenovis::GetInstance().getSession()->set_interactive(true);
    }

    void endTransform() {
        status = false;
        translate_matrix = glm::mat4(1.0f);
        scale_matrix = glm::mat4(1.0f);
        rotate_eular_vec = glm::vec3(0.0f);
        Zenovis::GetInstance().getSession()->set_interactive(false);
    }

    bool isTransforming() const {
        return status;
    }

    int getOperation() const {
        return operation;
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
    int operation;
};

}

#endif //__VIEWPORT_TRANSFORM_H__
