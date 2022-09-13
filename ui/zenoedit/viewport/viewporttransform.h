//
// Created by ryan on 22-8-8.
//

#ifndef __VIEWPORT_TRANSFORM_H__
#define __VIEWPORT_TRANSFORM_H__
#include "zenovis.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>

#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>
#include <zenomodel/include/nodesmgr.h>

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
        : m_objects_center(0.0f)
          , m_trans(0.0f)
          , m_scale(1.0f)
          , m_last_scale(1.0f)
          , m_rotate(0.0f)
          , m_status(false)
          , m_operation(NONE)
    {
    }

    FakeTransformer(const std::unordered_set<std::string>& names)
        : m_objects_center(0.0f)
          , m_trans(0.0f)
          , m_scale(1.0f)
          , m_last_scale(1.0f)
          , m_rotate(0.0f)
          , m_status(true)
          , m_operation(NONE)
    {
        addObject(names);
    }

    void addObject(const std::string& name) {
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        auto object = dynamic_cast<PrimitiveObject*>(scene->objectsMan->get(name).value());
        m_objects_center *= m_objects.size();
        auto& user_data = object->userData();
        zeno::vec3f bmin, bmax;
        if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
            bmin = user_data.getLiterial<zeno::vec3f>("_bboxMin");
            bmax = user_data.getLiterial<zeno::vec3f>("_bboxMax");
        } else {
            std::tie(bmin, bmax) = zeno::primBoundingBox(object);
            user_data.setLiterial("_bboxMin", bmin);
            user_data.setLiterial("_bboxMax", bmax);
        }
        if (!user_data.has("_translate")) {
            zeno::vec3f translate = {0, 0, 0};
            user_data.setLiterial("_translate", translate);
            zeno::vec3f rotate = {0, 0, 0};
            user_data.setLiterial("_rotate", rotate);
            zeno::vec3f scale = {1, 1, 1};
            user_data.setLiterial("_scale", scale);
        }
        auto m = zeno::vec_to_other<glm::vec3>(bmax);
        auto n = zeno::vec_to_other<glm::vec3>(bmin);
        m_objects_center += (m + n) / 2.0f;
        m_objects[name] = object;
        m_objects_center /= m_objects.size();
    }

    void addObject(const std::unordered_set<std::string>& names) {
        for (const auto& name : names) {
            addObject(name);
        }
    }

    void removeObject(const std::string& name) {
        auto p = m_objects.find(name);
        if (p == m_objects.end())
            return ;
        auto object = p->second;
        m_objects_center *= m_objects.size();
        auto& user_data = object->userData();
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
        m_objects_center -= (m + n) / 2.0f;
        m_objects.erase(p);
        m_objects_center /= m_objects.size();
    }

    void removeObject(const std::unordered_set<std::string>& names) {
        for (const auto& name : names) {
            removeObject(name);
        }
    }

    void translate(QVector3D start, QVector3D end, QVector3D axis) {
        auto diff = end - start;
        auto diff_vec3 = QVec3ToGLMVec3(diff);
        for (int i=0; i<3; i++)
            diff_vec3[i] *= axis[i];
        m_trans = diff_vec3;
        doTransform();
        m_operation = TRANSLATE;
    }

    void scale(QVector3D cur_vec, vec3i axis, float scale_size) {
        glm::vec3 cur_vec3 = QVec3ToGLMVec3(cur_vec);
        glm::vec3 scale(1.0f);
        for (int i=0; i<3; i++)
            if (axis[i] == 1) scale[i] = scale_size * 3;
        m_scale = scale;
        doTransform();
        m_operation = SCALE;
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
        m_rotate = angle * direct * axis_vec3;
        doTransform();
        m_operation = ROTATE;
    }

    void startTransform() {
        m_status = true;
        Zenovis::GetInstance().getSession()->set_interactive(true);
    }

    void endTransform(bool moved) {
        for (auto &[obj_name, obj] : m_objects) {
            auto& user_data = obj->userData();
            auto scale = user_data.getLiterial<zeno::vec3f>("_scale");
            for (int i=0; i<3; i++)
                scale[i] *= m_scale[i];
            user_data.setLiterial("_scale", scale);
        }

        if (moved) {
            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
            for (auto &[obj_name, obj] : m_objects) {
                QString node_id(obj_name.substr(0, obj_name.find_first_of(':')).c_str());
                auto search_result = pModel->search(node_id, SEARCH_NODEID);
                auto subgraph_index = search_result[0].subgIdx;
                auto node_index = search_result[0].targetIdx;
                auto inputs = node_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                if (node_id.contains("TransformPrimitive")  &&
                    inputs["translation"].linkIndice.empty() &&
                    inputs["eulerXYZ"].linkIndice.empty() &&
                    inputs["scaling"].linkIndice.empty()) {
                    syncToTransformNode(node_id, obj_name, pModel, node_index, subgraph_index);
                }
                else {
                    auto linked_transform_node_index =
                        linkedToVisibleTransformNode(node_id, node_index, pModel).value<QModelIndex>();
                    if (linked_transform_node_index.isValid()) {
                        auto linked_transform_node_id = linked_transform_node_index.data(ROLE_OBJID).toString();
                        syncToTransformNode(linked_transform_node_id, obj_name, pModel, linked_transform_node_index, subgraph_index);
                    }
                    else
                        createNewTransformNode(node_id, obj_name, pModel, node_index, subgraph_index);
                }
            }
        }

        m_status = false;
        m_operation = NONE;

        m_trans = {0, 0, 0};
        m_scale = {1, 1, 1};
        m_last_scale = {1, 1, 1};
        m_rotate = {0, 0, 0};

        Zenovis::GetInstance().getSession()->set_interactive(false);
    }

    bool isTransforming() const {
        return m_status;
    }

    int get_operation() const {
        return m_operation;
    }

    glm::vec3 getCenter() {
        return m_objects_center;
    }

    void clear() {
        m_objects.clear();
        m_trans = {0, 0, 0};
        m_scale = {1, 1, 1};
        m_last_scale = {1, 1, 1};
        m_rotate = {0, 0, 0};
        m_objects_center = {0, 0, 0};
    }

  private:
    static glm::vec3 QVec3ToGLMVec3(QVector3D QVec3) {
        return {QVec3.x(), QVec3.y(), QVec3.z()};
    }

    void print_mat4(std::string name, glm::mat4 mat) const {
        qDebug() << name.c_str() << ":";
        for (int i=0; i<4; i++) {
            qDebug() << mat[i][0] << " " << mat[i][1] << " " << mat[i][2] << " " << mat[i][3];
        }
    }

    template <class T>
    void print_vec3(std::string name, T& vec) {
        qDebug() << name.c_str() << ": " << vec[0] << " " << vec[1] << " " << vec[2];
    }

    void createNewTransformNode(QString& node_id, const std::string& obj_name, IGraphsModel* pModel, QModelIndex& node_index, QModelIndex& subgraph_index, bool change_visibility=true) {
        auto pos = node_index.data(ROLE_OBJPOS).toPointF();
        pos.setX(pos.x() + 10);
        auto new_node_id = NodesMgr::createNewNode(pModel, subgraph_index, "TransformPrimitive", pos);
        QString node_name = node_id.section('-', 1);
        QString out_sock = getNodePrimSockName(node_name.toStdString());
        EdgeInfo edge = {
            node_id,
            new_node_id,
            out_sock,
            "prim"
        };
        pModel->addLink(edge, subgraph_index, false);

        auto user_data = m_objects[obj_name]->userData();

        auto translate_vec3 = user_data.getLiterial<zeno::vec3f>("_translate");
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

        auto scaling_vec3 = user_data.getLiterial<zeno::vec3f>("_scale");
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
        auto rotate_vec3 = user_data.getLiterial<zeno::vec3f>("_rotate");
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

    QVariant linkedToVisibleTransformNode(QString& node_id, QModelIndex& node_index, IGraphsModel* pModel) {
        auto output_sockets = node_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        QString node_name = node_id.section("-", 1);
        std::string out_sock = getNodePrimSockName(node_name.toStdString());
        auto linked_edges = output_sockets[out_sock.c_str()].linkIndice;
        for (const auto& linked_edge : linked_edges) {
            auto next_node_id = linked_edge.data(ROLE_INNODE).toString();
            if (next_node_id.contains("TransformPrimitive")) {
                auto search_result = pModel->search(next_node_id, SEARCH_NODEID);
                auto linked_node_index = search_result[0].targetIdx;
                auto option = linked_node_index.data(ROLE_OPTIONS).toInt();
                if (option & OPT_VIEW) {
                    return QVariant::fromValue(linked_node_index);
                }
            }
        }
        return {};
    }

    void syncToTransformNode(QString& node_id, const std::string& obj_name, IGraphsModel* pModel, QModelIndex& node_index, QModelIndex& subgraph_index) {
        auto inputs = node_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        auto user_data = m_objects[obj_name]->userData();
        auto translate_old = inputs["translation"].info.defaultValue.value<UI_VECTYPE>();;
        auto translate_vec3 = user_data.getLiterial<zeno::vec3f>("_translate");
        QVector<double> translate_new = {
            translate_vec3[0],
            translate_vec3[1],
            translate_vec3[2]
        };
        PARAM_UPDATE_INFO translate_info = {
            "translation",
            QVariant::fromValue(translate_old),
            QVariant::fromValue(translate_new)
        };
        pModel->updateSocketDefl(node_id, translate_info, subgraph_index, true);
        // update scaling
        auto scaling_old = inputs["scaling"].info.defaultValue.value<UI_VECTYPE>();
        auto scaling_vec3 = user_data.getLiterial<zeno::vec3f>("_scale");
        QVector<double> scaling_new = {
            scaling_vec3[0],
            scaling_vec3[1],
            scaling_vec3[2]
        };
        PARAM_UPDATE_INFO scaling_info = {
            "scaling",
            QVariant::fromValue(scaling_old),
            QVariant::fromValue(scaling_new)
        };
        pModel->updateSocketDefl(node_id, scaling_info, subgraph_index, true);
        // update rotate
        auto rotate_old = inputs["eulerXYZ"].info.defaultValue.value<UI_VECTYPE>();
        auto rotate_vec3 = user_data.getLiterial<zeno::vec3f>("_rotate");
        QVector<double> rotate_new = {
            rotate_vec3[0],
            rotate_vec3[1],
            rotate_vec3[2]
        };
        PARAM_UPDATE_INFO rotate_info = {
            "eulerXYZ",
            QVariant::fromValue(rotate_old),
            QVariant::fromValue(rotate_new)
        };
        pModel->updateSocketDefl(node_id, rotate_info, subgraph_index, true);
    }

    void doTransform() {
        m_objects_center = {0, 0, 0};
        for (auto &[obj_name, obj] : m_objects) {
            auto& user_data = obj->userData();

            auto translate = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_translate"));
            auto rotate = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_rotate"));
            auto scale = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_scale"));
            auto pre_translate_matrix = glm::translate(translate);
            auto pre_rotateX_matrix = glm::rotate(rotate[0], glm::vec3({1, 0, 0}));
            auto pre_rotateY_matrix = glm::rotate(rotate[1], glm::vec3({0, 1, 0}));
            auto pre_rotateZ_matrix = glm::rotate(rotate[2], glm::vec3({0, 0, 1}));
            auto pre_scale_matrix = glm::scale(scale * m_last_scale);
            auto pre_transform_matrix = pre_translate_matrix * pre_rotateZ_matrix * pre_rotateY_matrix * pre_rotateX_matrix * pre_scale_matrix;
            auto inv_pre_transform = glm::inverse(pre_transform_matrix);

            translate += m_trans;
            // print_vec3("translate", translate);
            rotate += m_rotate;
            scale *= m_scale;
            print_vec3("scale", scale);
            m_last_scale = m_scale;

            auto translate_matrix = glm::translate(translate);
            // print_mat4("translate mat", translate_matrix);
            auto rotateX_matrix = glm::rotate(rotate[0], glm::vec3({1, 0, 0}));
            auto rotateY_matrix = glm::rotate(rotate[1], glm::vec3({0, 1, 0}));
            auto rotateZ_matrix = glm::rotate(rotate[2], glm::vec3({0, 0, 1}));
            auto rotate_matrix = rotateZ_matrix * rotateY_matrix * rotateX_matrix;
            // print_mat4("rotate mat", rotate_matrix);
            auto scale_matrix = glm::scale(scale);
            // print_mat4("scale mat", scale_matrix);
            auto transform_matrix = translate_matrix * rotate_matrix * scale_matrix * inv_pre_transform;
            // print_mat4("transform mat", transform_matrix);
            if (obj->has_attr("pos")) {
                // transform pos
                auto &pos = obj->attr<zeno::vec3f>("pos");
                for (auto &po : pos) {
                    auto p = zeno::vec_to_other<glm::vec3>(po);
                    auto t = transform_matrix * glm::vec4(p, 1.0f);
                    auto pt = glm::vec3(t) / t.w;
                    po = zeno::other_to_vec<3>(pt);
                }
            }
            if (obj->has_attr("nrm")) {
                // transform nrm
                auto &nrm = obj->attr<zeno::vec3f>("nrm");
                for (auto &vec : nrm) {
                    auto n = zeno::vec_to_other<glm::vec3>(vec);
                    glm::mat3 norm_matrix(transform_matrix);
                    norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                    auto t = glm::normalize(norm_matrix * n);
                    vec = zeno::other_to_vec<3>(t);
                }
            }
            vec3f bmin, bmax;
            if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
                std::tie(bmin, bmax) = primBoundingBox(obj);
                user_data.setLiterial("_bboxMin", bmin);
                user_data.setLiterial("_bboxMax", bmax);
            }
            user_data.setLiterial("_translate", zeno::other_to_vec<3>(translate));
            user_data.setLiterial("_rotate", zeno::other_to_vec<3>(rotate));
            m_objects_center += (zeno::vec_to_other<glm::vec3>(bmin) + zeno::vec_to_other<glm::vec3>(bmax)) / 2.0f;
        }
        m_objects_center /= m_objects.size();
    }

    const char* getNodePrimSockName(std::string node_name) {
        if (table.empty()) {
            table["TransformPrimitive"] = "outPrim";
            table["BindMaterial"] = "object";
        }
        if (table.find(node_name) == table.end())
            return "prim";
        return table[node_name].c_str();
    }

  private:
    std::unordered_map<std::string, PrimitiveObject*> m_objects;

    glm::vec3 m_objects_center;

    glm::vec3 m_trans;
    glm::vec3 m_scale;
    glm::vec3 m_last_scale;
    glm::vec3 m_rotate;

    bool m_status;
    int m_operation;

    std::unordered_map<std::string, std::string> table;
};

}

#endif //__VIEWPORT_TRANSFORM_H__
