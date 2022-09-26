#include "viewporttransform.h"

#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>
#include <zenomodel/include/nodesmgr.h>

#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace zeno {

FakeTransformer::FakeTransformer()
    : m_objects_center(0.0f)
      , m_trans(0.0f)
      , m_scale(1.0f)
      , m_rotate({0, 0, 0, 1})
      , m_last_trans(0.0f)
      , m_last_scale(1.0f)
      , m_last_rotate({0, 0, 0, 1})
      , m_status(false)
      , m_operation(NONE) {}

FakeTransformer::FakeTransformer(const std::unordered_set<std::string>& names)
    : m_objects_center(0.0f)
      , m_trans(0.0f)
      , m_scale(1.0f)
      , m_rotate({0, 0, 0, 1})
      , m_last_trans(0.0f)
      , m_last_scale(1.0f)
      , m_last_rotate({0, 0, 0, 1})
      , m_status(false)
      , m_operation(NONE)
{
    addObject(names);
}

void FakeTransformer::addObject(const std::string& name) {
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
        zeno::vec4f rotate = {0, 0, 0, 1};
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

void FakeTransformer::addObject(const std::unordered_set<std::string>& names) {
    for (const auto& name : names) {
        addObject(name);
    }
}

void FakeTransformer::removeObject(const std::string& name) {
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

void FakeTransformer::removeObject(const std::unordered_set<std::string>& names) {
    for (const auto& name : names) {
        removeObject(name);
    }
}

bool FakeTransformer::calcTransformStart(glm::vec3 ori, glm::vec3 dir, glm::vec3 front) {
    auto x_axis = glm::vec3(1, 0, 0);
    auto y_axis = glm::vec3(0, 1, 0);
    auto z_axis = glm::vec3(0, 0, 1);
    std::optional<glm::vec3> t;
    if (m_operation == TRANSLATE) {
        if (m_operation_mode == zenovis::INTERACT_X || m_operation_mode == zenovis::INTERACT_Y || m_operation_mode == zenovis::INTERACT_XY)
            t = hitOnPlane(ori, dir, z_axis, m_objects_center);
        else if (m_operation_mode == zenovis::INTERACT_Z || m_operation_mode == zenovis::INTERACT_XZ)
            t = hitOnPlane(ori, dir, y_axis, m_objects_center);
        else if (m_operation_mode == zenovis::INTERACT_YZ)
            t = hitOnPlane(ori, dir, x_axis, m_objects_center);
        else
            t = hitOnPlane(ori, dir, front, m_objects_center);
        if (t.has_value()) m_trans_start = t.value();
        else return false;
    }
    else if (m_operation == ROTATE) {
        if (m_operation_mode == zenovis::INTERACT_YZ)
            t = hitOnPlane(ori, dir, x_axis, m_objects_center);
        else if (m_operation_mode == zenovis::INTERACT_XZ)
            t = hitOnPlane(ori, dir, y_axis, m_objects_center);
        else if (m_operation_mode == zenovis::INTERACT_XY)
            t = hitOnPlane(ori, dir, z_axis, m_objects_center);
        else
            t = m_handler->getIntersect(ori, dir);
        if (t.has_value()) m_rotate_start = t.value();
        else return false;
    }
    return true;
}

bool FakeTransformer::clickedAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front) {
    if (!m_handler) return false;
    auto ray_ori = QVec3ToGLMVec3(ori);
    auto ray_dir = QVec3ToGLMVec3(dir);
    m_operation_mode = m_handler->collisionTest(ray_ori, ray_dir);
    if (!calcTransformStart(ray_ori, ray_dir, front)) return false;
    return m_operation_mode != zenovis::INTERACT_NONE;
}

/**
 * apply transform to all objects
 * @param camera_pos
 * @param mouse_pos
 * @param start_dir
 * @param end_dir
 * @param front
 */
void FakeTransformer::transform(QVector3D camera_pos, glm::vec2 mouse_pos, QVector3D ray_dir, glm::vec3 front, glm::mat4 vp) {
    if (m_operation == NONE) return;
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    auto ori = QVec3ToGLMVec3(camera_pos);
    auto dir = QVec3ToGLMVec3(ray_dir);

    auto x_axis = glm::vec3(1, 0, 0);
    auto y_axis = glm::vec3(0, 1, 0);
    auto z_axis = glm::vec3(0, 0, 1);

    if (m_operation == TRANSLATE) {
        if (m_operation_mode == zenovis::INTERACT_X) {
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), x_axis);
        }
        else if (m_operation_mode == zenovis::INTERACT_Y) {
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), y_axis);
        }
        else if (m_operation_mode == zenovis::INTERACT_Z) {
            auto cur_pos = hitOnPlane(ori, dir, y_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), z_axis);
        }
        else if (m_operation_mode == zenovis::INTERACT_XY) {
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 1, 0});
        }
        else if (m_operation_mode == zenovis::INTERACT_YZ) {
            auto cur_pos = hitOnPlane(ori, dir, x_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {0, 1, 1});
        }
        else if (m_operation_mode == zenovis::INTERACT_XZ) {
            auto cur_pos = hitOnPlane(ori, dir, y_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 0, 1});
        }
        else {
            auto cur_pos = hitOnPlane(ori, dir, front, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 1, 1});
        }
    }
    else if (m_operation == ROTATE) {
        if (m_operation_mode == zenovis::INTERACT_YZ) {
            auto cur_pos = hitOnPlane(ori, dir, x_axis, m_objects_center);
            if (cur_pos.has_value()) {
                auto start_vec = m_rotate_start - m_objects_center;
                auto end_vec = cur_pos.value() - m_objects_center;
                rotate(start_vec, end_vec, x_axis);
            }
        }
        else if (m_operation_mode == zenovis::INTERACT_XZ) {
            auto cur_pos = hitOnPlane(ori, dir, y_axis, m_objects_center);
            if (cur_pos.has_value()) {
                auto start_vec = m_rotate_start - m_objects_center;
                auto end_vec = cur_pos.value() - m_objects_center;
                rotate(start_vec, end_vec, y_axis);
            }
        }
        else if (m_operation_mode == zenovis::INTERACT_XY){
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value()) {
                auto start_vec = m_rotate_start - m_objects_center;
                auto end_vec = cur_pos.value() - m_objects_center;
                rotate(start_vec, end_vec, z_axis);
            }
        }
        else {
            auto start_vec = m_rotate_start - m_objects_center;
            auto test = m_handler->getIntersect(ori, dir);
            glm::vec3 end_vec;
            if (test.has_value())
                end_vec = test.value() - m_objects_center;
            else {
                auto p = hitOnPlane(ori, dir, front, m_objects_center);
                if (!p.has_value()) return;
                end_vec = p.value() - m_objects_center;
            }
            start_vec = glm::normalize(start_vec);
            end_vec = glm::normalize(end_vec);
            auto axis = glm::cross(start_vec, end_vec);
            rotate(start_vec, end_vec, axis);
        }
    }
    else if (m_operation == SCALE) {
        auto t_ctr = vp * glm::vec4(m_objects_center, 1.0f);
        glm::vec2 ctr = t_ctr / t_ctr[3];
        auto scale_size = glm::length(ctr - mouse_pos);
        if (m_operation_mode == zenovis::INTERACT_X) {
            scale(scale_size, {1, 0, 0});
        }
        else if (m_operation_mode == zenovis::INTERACT_Y) {
            scale(scale_size, {0, 1, 0});
        }
        else if (m_operation_mode == zenovis::INTERACT_Z) {
            scale(scale_size, {0, 0, 1});
        }
        else if (m_operation_mode == zenovis::INTERACT_XY) {
            scale(scale_size, {1, 1, 0});
        }
        else if (m_operation_mode == zenovis::INTERACT_YZ) {
            scale(scale_size, {0, 1, 1});
        }
        else if (m_operation_mode == zenovis::INTERACT_XZ) {
            scale(scale_size, {1, 0, 1});
        }
        else {
            scale(scale_size, {1, 1, 1});
        }
    }
}

void FakeTransformer::startTransform() {
    m_status = true;
    Zenovis::GetInstance().getSession()->set_interactive(true);
}

void FakeTransformer::endTransform(bool moved) {
    if (moved) {
        // write transform info to objects' user data
        for (auto &[obj_name, obj] : m_objects) {
            auto& user_data = obj->userData();

            if (m_operation == TRANSLATE) {
                auto trans = user_data.getLiterial<zeno::vec3f>("_translate");
                trans += other_to_vec<3>(m_trans);
                user_data.setLiterial("_translate", trans);
            }

            if (m_operation == ROTATE) {
                auto rotate = user_data.getLiterial<zeno::vec4f>("_rotate");
                auto pre_q = glm::quat(rotate[3], rotate[0], rotate[1], rotate[2]);
                auto dif_q = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
                auto res_q = glm::toQuat(glm::toMat4(dif_q) * glm::toMat4(pre_q));
                rotate = vec4f(res_q.x, res_q.y, res_q.z, res_q.w);
                user_data.setLiterial("_rotate", rotate);
            }

            if (m_operation == SCALE) {
                auto scale = user_data.getLiterial<zeno::vec3f>("_scale");
                for (int i = 0; i < 3; i++)
                    scale[i] *= m_scale[i];
                user_data.setLiterial("_scale", scale);
            }
        }

        // sync to node system
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        for (auto &[obj_name, obj] : m_objects) {
            QString node_id(obj_name.substr(0, obj_name.find_first_of(':')).c_str());
            auto search_result = pModel->search(node_id, SEARCH_NODEID);
            auto subgraph_index = search_result[0].subgIdx;
            auto node_index = search_result[0].targetIdx;
            auto inputs = node_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (node_id.contains("TransformPrimitive")  &&
                inputs["translation"].linkIndice.empty() &&
                inputs["quatRotation"].linkIndice.empty() &&
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

    m_trans = {0, 0, 0};
    m_scale = {1, 1, 1};
    m_rotate = {0, 0, 0, 1};

    m_last_trans = {0, 0, 0};
    m_last_scale = {1, 1, 1};
    m_last_rotate = {0, 0, 0, 1};

    m_operation_mode = zenovis::INTERACT_NONE;
    m_handler->setMode(zenovis::INTERACT_NONE);

    Zenovis::GetInstance().getSession()->set_interactive(false);
}

bool FakeTransformer::isTransforming() const {
    return m_status;
}

void FakeTransformer::toTranslate() {
    if (m_objects.empty()) return;
    if (m_operation == TRANSLATE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = TRANSLATE;
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        m_handler = zenovis::makeTransHandler(scene,zeno::other_to_vec<3>(m_objects_center));
    }
    Zenovis::GetInstance().getSession()->set_handler(m_handler);
}

void FakeTransformer::toRotate() {
    if (m_objects.empty()) return;
    if (m_operation == ROTATE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = ROTATE;
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(m_objects_center));
    }
    Zenovis::GetInstance().getSession()->set_handler(m_handler);
}

void FakeTransformer::toScale() {
    if (m_objects.empty()) return;
    if (m_operation == SCALE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = SCALE;
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        m_handler = zenovis::makeScaleHandler(scene,zeno::other_to_vec<3>(m_objects_center));
    }
    Zenovis::GetInstance().getSession()->set_handler(m_handler);
}

void FakeTransformer::changeCoordSys() {
    if (m_coord_sys == zenovis::VIEW_COORD_SYS)
        m_coord_sys = zenovis::WORLD_COORD_SYS;
    else
        ++m_coord_sys;
    if (m_handler)
        m_handler->setCoordSys(m_coord_sys);
}

bool FakeTransformer::isTransformMode() const {
    return m_operation != NONE;
}

glm::vec3 FakeTransformer::getCenter() const {
    return m_objects_center;
}

void FakeTransformer::clear() {
    m_objects.clear();
    m_trans = {0, 0, 0};
    m_scale = {1, 1, 1};
    m_rotate = {0, 0, 0, 1};
    m_last_trans = {0, 0, 0};
    m_last_scale = {1, 1, 1};
    m_last_rotate = {0, 0, 0, 1};
    m_operation = NONE;
    m_handler = nullptr;
    Zenovis::GetInstance().getSession()->set_handler(m_handler);
    m_objects_center = {0, 0, 0};
}

void FakeTransformer::translate(glm::vec3 start, glm::vec3 end, glm::vec3 axis) {
    auto diff = end - start;
    diff *= axis;
    m_trans = diff;
    doTransform();
}

void FakeTransformer::scale(float scale_size, vec3i axis) {
    glm::vec3 scale(1.0f);
    for (int i=0; i<3; i++)
        if (axis[i] == 1) scale[i] = fmax(scale_size * 3, 0.1);
    m_scale = scale;
    doTransform();
}

void FakeTransformer::rotate(glm::vec3 start_vec, glm::vec3 end_vec, glm::vec3 axis) {
    start_vec = glm::normalize(start_vec);
    end_vec = glm::normalize(end_vec);
    if (glm::length(start_vec - end_vec) < 0.0001) return;
    auto cross_vec = glm::cross(start_vec, end_vec);
    float direct = 1.0f;
    if (glm::dot(cross_vec, axis) < 0)
        direct = -1.0f;
    float angle = acos(fmin(fmax(glm::dot(start_vec, end_vec), -1.0f), 1.0f));
    glm::quat q(glm::rotate(angle * direct, axis));
    m_rotate = {q.x, q.y, q.z, q.w};
    doTransform();
}

void FakeTransformer::createNewTransformNode(QString& node_id, const std::string& obj_name, IGraphsModel* pModel,
                                             QModelIndex& node_index, QModelIndex& subgraph_index, bool change_visibility) {
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
    auto rotate_vec4 = user_data.getLiterial<zeno::vec4f>("_rotate");
    QVector<double> rotate = {
        rotate_vec4[0],
        rotate_vec4[1],
        rotate_vec4[2],
        rotate_vec4[3]
    };
    PARAM_UPDATE_INFO rotate_info = {
        "quatRotation",
        QVariant::fromValue(QVector<double>{0, 0, 0, 1}),
        QVariant::fromValue(rotate)
    };
    pModel->updateSocketDefl(new_node_id, rotate_info, subgraph_index, true);

    if (1) {
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

QVariant FakeTransformer::linkedToVisibleTransformNode(QString& node_id, QModelIndex& node_index, IGraphsModel* pModel) {
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

void FakeTransformer::syncToTransformNode(QString& node_id, const std::string& obj_name, IGraphsModel* pModel,
                         QModelIndex& node_index, QModelIndex& subgraph_index) {
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
    auto rotate_old = inputs["quatRotation"].info.defaultValue.value<UI_VECTYPE>();
    auto rotate_vec4 = user_data.getLiterial<zeno::vec4f>("_rotate");
    QVector<double> rotate_new = {
        rotate_vec4[0],
        rotate_vec4[1],
        rotate_vec4[2],
        rotate_vec4[3]
    };
    PARAM_UPDATE_INFO rotate_info = {
        "quatRotation",
        QVariant::fromValue(rotate_old),
        QVariant::fromValue(rotate_new)
    };
    pModel->updateSocketDefl(node_id, rotate_info, subgraph_index, true);
}

void FakeTransformer::doTransform() {
    m_objects_center = {0, 0, 0};
    for (auto &[obj_name, obj] : m_objects) {
        auto& user_data = obj->userData();

        // get transform info
        auto translate = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_translate"));
        auto rotate = zeno::vec_to_other<glm::vec4>(user_data.getLiterial<zeno::vec4f>("_rotate"));
        auto scale = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_scale"));

        // inv last transform
        auto pre_translate_matrix = glm::translate(translate + m_last_trans);
        auto pre_quaternion = glm::quat(rotate[3], rotate[0], rotate[1], rotate[2]);
        auto last_quaternion = glm::quat(m_last_rotate[3], m_last_rotate[0], m_last_rotate[1], m_last_rotate[2]);
        auto pre_rotate_matrix = glm::toMat4(pre_quaternion);
        auto pre_scale_matrix = glm::scale(scale * m_last_scale);
        auto pre_transform_matrix = pre_translate_matrix * glm::toMat4(last_quaternion) * pre_rotate_matrix * pre_scale_matrix;
        auto inv_pre_transform = glm::inverse(pre_transform_matrix);

        // do this transform
        auto translate_matrix = glm::translate(translate + m_trans);
        auto cur_quaternion = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
        auto rotate_matrix = glm::toMat4(cur_quaternion) * pre_rotate_matrix;
        auto scale_matrix = glm::scale(scale * m_scale);
        auto transform_matrix = translate_matrix * rotate_matrix * scale_matrix * inv_pre_transform;

        m_last_trans = m_trans;
        m_last_rotate = m_rotate;
        m_last_scale = m_scale;

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
        m_objects_center += (zeno::vec_to_other<glm::vec3>(bmin) + zeno::vec_to_other<glm::vec3>(bmax)) / 2.0f;
    }
    m_objects_center /= m_objects.size();
    m_handler->setCenter({m_objects_center[0], m_objects_center[1], m_objects_center[2]});
}

const char* FakeTransformer::getNodePrimSockName(std::string node_name) {
    if (table.empty()) {
        table["TransformPrimitive"] = "outPrim";
        table["BindMaterial"] = "object";
    }
    if (table.find(node_name) == table.end())
        return "prim";
    return table[node_name].c_str();
}

}