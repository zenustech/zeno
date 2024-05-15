#include "transform.h"

#include <zeno/funcs/PrimitiveTools.h>
#include <zenovis/ObjectsManager.h>
#include <zenomodel/include/uihelper.h>
#include "zenomainwindow.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace zeno {

FakeTransformer::FakeTransformer(ViewportWidget* viewport)
    : m_viewport(viewport)
{
}

zenovis::Scene* FakeTransformer::scene() const
{
    ZASSERT_EXIT(m_viewport, nullptr);
    auto session = m_viewport->getSession();
    ZASSERT_EXIT(session, nullptr);
    auto pScene = session->get_scene();
    return pScene;
}

zenovis::Session* FakeTransformer::session() const
{
    ZASSERT_EXIT(m_viewport, nullptr);
    auto session = m_viewport->getSession();
    return session;
}

void FakeTransformer::addObject(const std::string& name) {
    if (name.empty()) return;

    auto pZenovis = m_viewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);
    auto sess = pZenovis->getSession();
    ZASSERT_EXIT(sess);
    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);
    if (!scene->objectsMan->get(name).has_value())
        return;

    auto object = dynamic_cast<PrimitiveObject*>(scene->objectsMan->get(name).value());
    m_objects[name] = object;
    createNewTransformNodeNameWhenMissing(name);
    auto& node_sync = NodeSyncMgr::GetInstance();
    auto prim_node_location = node_sync.searchNodeOfPrim(name);
    auto prim_node = prim_node_location->node;
    if (!node_sync.checkNodeType(prim_node, "PrimitiveTransform")) {
        // prim comes from another type node
        prim_node = node_sync.checkNodeLinkedSpecificNode(prim_node, "PrimitiveTransform")->node;
    }

    m_init_pivot = node_sync.getInputValVec3(prim_node, "pivotPos");
    m_init_localXOrg = glm::normalize(node_sync.getInputValVec3(prim_node, "localX"));
    m_init_localYOrg = glm::normalize(node_sync.getInputValVec3(prim_node, "localY"));
    m_transaction_start_translation = node_sync.getInputValVec3(prim_node, "translation");
    m_transaction_start_scaling = node_sync.getInputValVec3(prim_node, "scaling");
    m_transaction_start_rotation = node_sync.getInputValQuat(prim_node, "quatRotation");

    {
        auto translate_matrix = glm::translate(m_transaction_start_translation);
        auto rotate_matrix = glm::toMat4(m_transaction_start_rotation);
        auto scale_matrix = glm::scale(m_transaction_start_scaling);
        last_transform_matrix = get_pivot_to_world() * translate_matrix *  rotate_matrix * scale_matrix * get_pivot_to_local();
    }
}

void FakeTransformer::addObjects(const std::unordered_set<std::string>& names) {
    this->clear();
    for (const auto& name : names) {
        addObject(name);
    }
}

bool FakeTransformer::calcTransformStart(glm::vec3 ori, glm::vec3 dir, glm::vec3 front) {
    auto x_axis = get_cur_self_X();
    auto y_axis = get_cur_self_Y();
    auto z_axis = glm::cross(x_axis, y_axis);
    std::optional<glm::vec3> t;
    if (m_operation == TransOpt::TRANSLATE) {
        if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_X || m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Y || m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XY)
            t = hitOnPlane(ori, dir, z_axis, get_cur_self_center());
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Z || m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XZ)
            t = hitOnPlane(ori, dir, y_axis, get_cur_self_center());
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_YZ)
            t = hitOnPlane(ori, dir, x_axis, get_cur_self_center());
        else
            t = hitOnPlane(ori, dir, front, get_cur_self_center());
        if (t.has_value()) m_move_start = t.value();
        else return false;
    }
    else if (m_operation == TransOpt::ROTATE) {
        if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_YZ)
            t = hitOnPlane(ori, dir, x_axis, get_cur_self_center());
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XZ)
            t = hitOnPlane(ori, dir, y_axis, get_cur_self_center());
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XY)
            t = hitOnPlane(ori, dir, z_axis, get_cur_self_center());
        else
            t = m_handler->getIntersect(ori, dir);
        if (t.has_value()) m_move_start = t.value();
        else return false;
    }
    return true;
}

bool FakeTransformer::clickedAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front) {
    if (!m_handler) return false;
    auto ray_ori = QVec3ToGLMVec3(ori);
    auto ray_dir = QVec3ToGLMVec3(dir);
    m_operation_mode = m_handler->handleClick(ray_ori, ray_dir);
    if (!calcTransformStart(ray_ori, ray_dir, front)) return false;
    return m_operation_mode != zenovis::OPERATION_MODE::INTERACT_NONE;
}

bool FakeTransformer::hoveredAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front)
{
    if (!m_handler) return false;
    auto ray_ori = QVec3ToGLMVec3(ori);
    auto ray_dir = QVec3ToGLMVec3(dir);
    zenovis::OPERATION_MODE mode = m_handler->handleHover(ray_ori, ray_dir);
    if (!calcTransformStart(ray_ori, ray_dir, front)) return false;
    return mode != zenovis::OPERATION_MODE::INTERACT_NONE;
}

void FakeTransformer::transform(QVector3D camera_pos, QVector3D ray_dir, glm::vec2 mouse_start, glm::vec2 mouse_pos, glm::vec3 front, glm::mat4 vp) {
    if (m_operation == TransOpt::NONE) return;

    auto pZenovis = m_viewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);
    auto sess = pZenovis->getSession();
    ZASSERT_EXIT(sess);
    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);

    auto ori = QVec3ToGLMVec3(camera_pos);
    auto dir = QVec3ToGLMVec3(ray_dir);

    auto x_axis = glm::vec3(1, 0, 0);
    auto y_axis = glm::vec3(0, 1, 0);
    auto z_axis = glm::vec3(0, 0, 1);

    auto localX = get_transaction_start_X();
    auto localY = get_transaction_start_Y();
    auto localZ = glm::cross(localX, localY);
    auto cur_to_world = glm::mat3(1);
    cur_to_world[0] = localX;
    cur_to_world[1] = localY;
    cur_to_world[2] = localZ;
    auto cur_to_local = glm::inverse(cur_to_world);

    auto pivot_to_local = get_pivot_to_local();

    if (m_operation == TransOpt::TRANSLATE) {
        if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_X) {
            auto cur_pos = hitOnPlane(ori, dir, localZ, get_cur_self_center());
            if (cur_pos.has_value()) {
                translate(m_move_start, cur_pos.value(), x_axis, cur_to_local, cur_to_world, pivot_to_local);
            }
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Y) {
            auto cur_pos = hitOnPlane(ori, dir, localZ, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), y_axis, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Z) {
            auto cur_pos = hitOnPlane(ori, dir, localY, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), z_axis, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XY) {
            auto cur_pos = hitOnPlane(ori, dir, localZ, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), {1, 1, 0}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_YZ) {
            auto cur_pos = hitOnPlane(ori, dir, localX, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), {0, 1, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XZ) {
            auto cur_pos = hitOnPlane(ori, dir, localY, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), {1, 0, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else {
            auto cur_pos = hitOnPlane(ori, dir, front, get_cur_self_center());
            if (cur_pos.has_value())
                translate(m_move_start, cur_pos.value(), {1, 1, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
    }
    else if (m_operation == TransOpt::ROTATE) {
        if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_YZ) {
            auto cur_pos = hitOnPlane(ori, dir, localX, get_cur_self_center());
            if (cur_pos.has_value()) {
                auto start_vec = m_move_start - get_cur_self_center();
                auto end_vec = cur_pos.value() - get_cur_self_center();
                rotate(start_vec, end_vec, localX);
            }
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XZ) {
            auto cur_pos = hitOnPlane(ori, dir, localY, get_cur_self_center());
            if (cur_pos.has_value()) {
                auto start_vec = m_move_start - get_cur_self_center();
                auto end_vec = cur_pos.value() - get_cur_self_center();
                rotate(start_vec, end_vec, localY);
            }
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XY){
            auto cur_pos = hitOnPlane(ori, dir, localZ, get_cur_self_center());
            if (cur_pos.has_value()) {
                auto start_vec = m_move_start - get_cur_self_center();
                auto end_vec = cur_pos.value() - get_cur_self_center();
                rotate(start_vec, end_vec, localZ);
            }
        }
        else {
            auto start_vec = m_move_start - get_cur_self_center();
            auto test = m_handler->getIntersect(ori, dir);
            glm::vec3 end_vec;
            if (test.has_value())
                end_vec = test.value() - get_cur_self_center();
            else {
                auto p = hitOnPlane(ori, dir, front, get_cur_self_center());
                if (!p.has_value()) return;
                end_vec = p.value() - get_cur_self_center();
            }
            start_vec = glm::normalize(start_vec);
            end_vec = glm::normalize(end_vec);
            auto axis = glm::cross(start_vec, end_vec);
            rotate(start_vec, end_vec, axis);
        }
    }
    else if (m_operation == TransOpt::SCALE) {
        // make a circle, center is get_cur_self_center()
        // when mouse press, get the circle's radius, r = len(get_cur_self_center() - mouse_start)
        // when mouse move, get the len from center to mouse_pos, d = len(get_cur_self_center() - mouse_pos)
        // so the scale is d / r
        auto t_ctr = vp * glm::vec4(get_cur_self_center(), 1.0f);
        glm::vec2 ctr = t_ctr / t_ctr[3];
        auto len_ctr_start = glm::length(ctr - mouse_start);
        if (len_ctr_start < 0.001) return;

        auto len_ctr_pos = glm::length(ctr - mouse_pos);
        auto scale_size = len_ctr_pos / len_ctr_start;
        if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_X) {
            scale(scale_size, {1, 0, 0});
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Y) {
            scale(scale_size, {0, 1, 0});
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_Z) {
            scale(scale_size, {0, 0, 1});
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XY) {
            scale(scale_size, {1, 1, 0});
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_YZ) {
            scale(scale_size, {0, 1, 1});
        }
        else if (m_operation_mode == zenovis::OPERATION_MODE::INTERACT_XZ) {
            scale(scale_size, {1, 0, 1});
        }
        else {
            scale(scale_size, {1, 1, 1});
        }
    }
}

bool FakeTransformer::isTransforming() const {
    return m_isTransforming;
}

void FakeTransformer::startTransform() {
    markObjectsInteractive();
}

void FakeTransformer::createNewTransformNode(NodeLocation& node_location, const std::string& obj_name) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto out_sock = node_sync.getPrimSockName(node_location.node);
    auto new_node_location = node_sync.generateNewNode(node_location,
                                                       "PrimitiveTransform",
                                                       out_sock,
                                                       "prim");

    // make node not visible
    node_sync.updateNodeVisibility(node_location);
    // make new node visible
    node_sync.updateNodeVisibility(new_node_location.value());
    if (m_objects.count(obj_name)) {
        zeno::vec3f centroid = {};
        auto prim = m_objects[obj_name];
        for (auto i = 0; i < prim->verts.size(); i++) {
            centroid += prim->verts[i];
        }
        centroid /= prim->verts.size();
        QVector<double> pivotPos = {
            centroid[0],
            centroid[1],
            centroid[2]
        };
        node_sync.updateNodeInputVec(new_node_location.value(), "pivotPos", pivotPos);
        node_sync.updateNodeInputString(new_node_location.value(), "pivot", "custom");
    }
}

void FakeTransformer::createNewTransformNodeNameWhenMissing(std::string const &node_name) {
    auto& node_sync = NodeSyncMgr::GetInstance();
    auto prim_node_location = node_sync.searchNodeOfPrim(node_name);
    auto& prim_node = prim_node_location->node;
    if (!node_sync.checkNodeType(prim_node, "PrimitiveTransform")) {
        auto linked_transform_node = node_sync.checkNodeLinkedSpecificNode(prim_node, "PrimitiveTransform");
        if (!linked_transform_node.has_value()) {
            createNewTransformNode(prim_node_location.value(), node_name);
        }
    }
}

void FakeTransformer::syncToTransformNode(NodeLocation& node_location, const std::string& obj_name) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto translate_data = m_transaction_start_translation + _transaction_trans;
    QVector<double> translate = {
        translate_data[0],
        translate_data[1],
        translate_data[2]
    };
    node_sync.updateNodeInputVec(node_location,
                                 "translation",
                                 translate);
    // update scaling
    auto scaling_data = m_transaction_start_scaling;
    for (int i = 0; i < 3; i++) {
        scaling_data[i] *= _transaction_scale[i];
    }
    QVector<double> scaling = {
        scaling_data[0],
        scaling_data[1],
        scaling_data[2]
    };
    node_sync.updateNodeInputVec(node_location,
                                 "scaling",
                                 scaling);
    // update rotate
    auto pre_q = m_transaction_start_rotation;
    auto dif_q = _transaction_rotate;
    auto res_q = glm::toQuat(glm::toMat4(dif_q) * glm::toMat4(pre_q));
    auto rotate_data = vec4f(res_q.x, res_q.y, res_q.z, res_q.w);
    QVector<double> rotate = {
        rotate_data[0],
        rotate_data[1],
        rotate_data[2],
        rotate_data[3]
    };
    node_sync.updateNodeInputVec(node_location,
                                 "quatRotation",
                                 rotate);
}

void FakeTransformer::endTransform(bool moved) {
    if (moved) {
        // write transform info to objects' user data
        for (auto &[obj_name, obj] : m_objects) {
            if (m_operation == TransOpt::TRANSLATE) {
                m_transaction_start_translation += _transaction_trans;
            }

            if (m_operation == TransOpt::ROTATE) {
                auto pre_q = m_transaction_start_rotation;
                auto dif_q = _transaction_rotate;
                m_transaction_start_rotation = glm::toQuat(glm::toMat4(dif_q) * glm::toMat4(pre_q));
            }

            if (m_operation == TransOpt::SCALE) {
                for (int i = 0; i < 3; i++)
                    m_transaction_start_scaling[i] *= _transaction_scale[i];
            }
        }
    }
    unmarkObjectsInteractive();

    _transaction_trans = {0, 0, 0};
    _transaction_scale = {1, 1, 1};
    _transaction_rotate = {1, 0, 0, 0};

    m_operation_mode = zenovis::OPERATION_MODE::INTERACT_NONE;
    m_handler->setMode(zenovis::OPERATION_MODE::INTERACT_NONE);
}

void FakeTransformer::toTranslate() {
    if (m_operation == TransOpt::TRANSLATE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        auto session = this->session();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        addObjects(scene->selected);
        if (m_objects.empty()) return;
        m_operation = TransOpt::TRANSLATE;
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        session->set_handler(m_handler);
    }
}

void FakeTransformer::toRotate() {
    if (m_operation == TransOpt::ROTATE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        auto session = this->session();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        addObjects(scene->selected);
        if (m_objects.empty()) return;
        m_operation = TransOpt::ROTATE;
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        session->set_handler(m_handler);
    }
}

void FakeTransformer::toScale() {
    if (m_operation == TransOpt::SCALE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        auto session = this->session();
        ZASSERT_EXIT(session);
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        addObjects(scene->selected);
        if (m_objects.empty()) return;
        m_operation = TransOpt::SCALE;
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        session->set_handler(m_handler);
    }
}

void FakeTransformer::markObjectInteractive(const std::string& obj_name) {
    auto& user_data = m_objects[obj_name]->userData();
    user_data.setLiterial("interactive", 1);
}

void FakeTransformer::unmarkObjectInteractive(const std::string& obj_name) {
    auto& user_data = m_objects[obj_name]->userData();
    user_data.setLiterial("interactive", 0);
}

void FakeTransformer::markObjectsInteractive() {
    m_isTransforming = true;
    for (const auto& [obj_name, obj] : m_objects) {
        markObjectInteractive(obj_name);
    }
}

void FakeTransformer::unmarkObjectsInteractive() {
    m_isTransforming = false;
    for (const auto& [obj_name, obj] : m_objects) {
        unmarkObjectInteractive(obj_name);
    }
}

void FakeTransformer::resizeHandler(int dir) {
    if (!m_handler) return;
    switch (dir) {
    case 0:
        m_handler_scale = 1.f;
        break;
    case 1:
        m_handler_scale /= 0.89;
        break;
    case 2:
        m_handler_scale *= 0.89;
        break;
    default:
        break;
    }
    m_handler->resize(m_handler_scale);
}

void FakeTransformer::changeTransOpt() {
    if (m_objects.empty()) return;
    if (m_operation == TransOpt::SCALE)
        m_operation = TransOpt::NONE;
    else if (m_operation == TransOpt::NONE)
        m_operation = TransOpt::TRANSLATE;
    else if (m_operation == TransOpt::TRANSLATE)
        m_operation = TransOpt::ROTATE;
    else if (m_operation == TransOpt::ROTATE)
        m_operation = TransOpt::SCALE;

    auto session = this->session();
    ZASSERT_EXIT(session);
    auto scene = this->scene();

    switch (m_operation) {
    case TransOpt::TRANSLATE:
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        break;
    case TransOpt::ROTATE:
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        break;
    case TransOpt::SCALE:
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(get_cur_self_center()), zeno::other_to_vec<3>(get_cur_self_X()), zeno::other_to_vec<3>(get_cur_self_Y()), m_handler_scale);
        break;
    case TransOpt::NONE:
        m_handler = nullptr;
    default:
        break;
    }
    session->set_handler(m_handler);
}

void FakeTransformer::changeCoordSys() {
    if (m_coord_sys == zenovis::COORD_SYS::VIEW_COORD_SYS)
        m_coord_sys = zenovis::COORD_SYS::WORLD_COORD_SYS;
    else if (m_coord_sys == zenovis::COORD_SYS::WORLD_COORD_SYS)
        m_coord_sys = zenovis::COORD_SYS::LOCAL_COORD_SYS;
    else if (m_coord_sys == zenovis::COORD_SYS::LOCAL_COORD_SYS)
        m_coord_sys = zenovis::COORD_SYS::VIEW_COORD_SYS;
    if (m_handler)
        m_handler->setCoordSys(m_coord_sys);
}

TransOpt FakeTransformer::getTransOpt() {
    return m_operation;
}

void FakeTransformer::setTransOpt(TransOpt opt) {
    m_operation = opt;
}

bool FakeTransformer::isTransformMode() const {
    return m_operation != TransOpt::NONE;
}

void FakeTransformer::clear() {
    m_objects.clear();
    _transaction_trans = {0, 0, 0};
    _transaction_scale = {1, 1, 1};
    _transaction_rotate = {1, 0, 0, 0};
    m_operation = TransOpt::NONE;
    m_handler = nullptr;

    auto session = this->session();
    ZASSERT_EXIT(session);
    session->set_handler(m_handler);
}

void FakeTransformer::translate(glm::vec3 start, glm::vec3 end, glm::vec3 axis, glm::mat3 to_local, glm::mat3 to_world, glm::mat3 org_to_local) {
    auto diff = end - start; // diff in world space
    diff = to_local * diff; // diff in cur local coord
    diff *= axis;
    diff = to_world * diff; // diff in world space
    diff = org_to_local * diff; // diff in pivot local space
    _transaction_trans = diff;
    doTransform();
}

void FakeTransformer::scale(float scale_size, vec3i axis) {
    glm::vec3 scale(1.0f);
    for (int i = 0; i < 3; i++)
        if (axis[i] == 1) scale[i] = std::max(scale_size, 0.1f);
    _transaction_scale = scale;
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
    float angle = acos(zeno::clamp(glm::dot(start_vec, end_vec), -1.0f, 1.0f));
    axis = glm::mat3(get_pivot_to_local()) * axis;
    _transaction_rotate = glm::quat(glm::rotate(angle * direct, axis));
    doTransform();
}

void FakeTransformer::doTransform() {
    auto pivot_to_world = get_pivot_to_world();
    auto pivot_to_local = get_pivot_to_local();

    for (auto &[obj_name, obj] : m_objects) {
        auto& node_sync = NodeSyncMgr::GetInstance();
        auto prim_node_location = node_sync.searchNodeOfPrim(obj_name);
        auto prim_node = prim_node_location->node;
        if (!node_sync.checkNodeType(prim_node, "PrimitiveTransform")) {
            // prim comes from another type node
            prim_node = node_sync.checkNodeLinkedSpecificNode(prim_node, "PrimitiveTransform")->node;
        }

        // do this transform
        auto translate_matrix = glm::translate(m_transaction_start_translation + _transaction_trans);
        auto cur_quaternion = _transaction_rotate;
        auto rotate_matrix = glm::toMat4(cur_quaternion) * glm::toMat4(m_transaction_start_rotation);
        auto scale_matrix = glm::scale(m_transaction_start_scaling * _transaction_scale);
        auto transform_matrix = pivot_to_world *  translate_matrix *  rotate_matrix * scale_matrix * pivot_to_local;

        {
            // transform pos
            auto &pos = obj->attr<zeno::vec3f>("pos");
#pragma omp parallel for
            // for (auto &po : pos) {
            for (auto i = 0; i < pos.size(); ++i) {
                auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                auto t = transform_matrix * glm::inverse(last_transform_matrix) * glm::vec4(p, 1.0f);
                auto pt = glm::vec3(t) / t.w;
                pos[i] = zeno::other_to_vec<3>(pt);
            }
        }
        if (obj->has_attr("nrm")) {
            // transform nrm
            auto &nrm = obj->attr<zeno::vec3f>("nrm");
#pragma omp parallel for
            // for (auto &vec : nrm) {
            for (auto i = 0; i < nrm.size(); ++i) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                glm::mat3 norm_matrix(transform_matrix * glm::inverse(last_transform_matrix));
                norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                auto t = glm::normalize(norm_matrix * n);
                nrm[i] = zeno::other_to_vec<3>(t);
            }
        }
        last_transform_matrix = transform_matrix;
    }
    m_handler->setCenter(other_to_vec<3>(get_cur_self_center()), other_to_vec<3>(get_cur_self_X()), other_to_vec<3>(get_cur_self_Y()));
    // sync to panel
    {
        // sync to node system
        zeno::scope_exit sp([] {
            IGraphsModel *pGraphs = zenoApp->graphsManagment()->currentModel();
            if (pGraphs)
                pGraphs->setApiRunningEnable(true);
        });
        //only update nodes.
        IGraphsModel *pGraphs = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphs);
        pGraphs->setApiRunningEnable(false);

        for (auto &[obj_name, obj] : m_objects) {
            auto& node_sync = NodeSyncMgr::GetInstance();
            auto prim_node_location = node_sync.searchNodeOfPrim(obj_name);
            auto& prim_node = prim_node_location->node;
            if (node_sync.checkNodeType(prim_node, "PrimitiveTransform")) {
                syncToTransformNode(prim_node_location.value(), obj_name);
            }
            else {
                // prim comes from another type node
                auto linked_transform_node = node_sync.checkNodeLinkedSpecificNode(prim_node, "PrimitiveTransform");
                syncToTransformNode(linked_transform_node.value(), obj_name);
            }
        }

    }
}

}