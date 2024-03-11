#include "transform.h"

#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zenovis/ObjectsManager.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/uihelper.h>
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace zeno {

FakeTransformer::FakeTransformer(ViewportWidget* viewport)
    : m_objects_center(0.0f)
      , m_pivot(0.0f)
      , m_localXOrg(1, 0, 0)
      , m_localYOrg(0, 1, 0)
      , m_localX(1, 0, 0)
      , m_localY(0, 1, 0)
      , m_trans(0.0f)
      , m_scale(1.0f)
      , m_rotate({0, 0, 0, 1})
      , m_status(false)
      , m_operation(TransOpt::NONE)
      , m_handler_scale(1.f)
      , m_viewport(viewport)
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
    if (!user_data.has("_pivot")) {
        zeno::vec3f translate = {0, 0, 0};
        user_data.setLiterial("_translate", translate);
        zeno::vec4f rotate = {0, 0, 0, 1};
        user_data.setLiterial("_rotate", rotate);
        zeno::vec3f scale = {1, 1, 1};
        user_data.setLiterial("_scale", scale);
        auto bboxCenter = (bmin + bmax) / 2;
        user_data.set2("_pivot", bboxCenter);
        user_data.set2("_localX", vec3f(1, 0, 0));
        user_data.set2("_localY", vec3f(0, 1, 0));
        if (object->has_attr("pos") && !object->has_attr("_origin_pos")) {
            auto &pos = object->attr<zeno::vec3f>("pos");
            object->verts.add_attr<zeno::vec3f>("_origin_pos") = pos;
        }
        if (object->has_attr("nrm") && !object->has_attr("_origin_nrm")) {
            auto &nrm = object->attr<zeno::vec3f>("nrm");
            object->verts.add_attr<zeno::vec3f>("_origin_nrm") = nrm;
        }
    }
    auto m = zeno::vec_to_other<glm::vec3>(bmax);
    auto n = zeno::vec_to_other<glm::vec3>(bmin);
    m_pivot = zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_pivot"));
    m_localXOrg = zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_localX"));
    m_localYOrg = zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_localY"));
    auto rot = user_data.get2<vec4f>("_rotate");
    auto quat = glm::quat(rot[3], rot[0], rot[1], rot[2]);
    m_localX = glm::toMat3(quat) * m_localXOrg;
    m_localY = glm::toMat3(quat) * m_localYOrg;

    auto pivot_to_world = glm::mat3(1);
    pivot_to_world[0] = m_localXOrg;
    pivot_to_world[1] = m_localYOrg;
    pivot_to_world[2] = glm::cross(m_localXOrg, m_localYOrg);

    m_objects_center *= m_objects.size();
    m_objects_center += m_pivot + pivot_to_world * zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_translate"));
    m_objects[name] = object;
    m_objects_center /= m_objects.size();
}

void FakeTransformer::addObjects(const std::unordered_set<std::string>& names) {
    this->clear();
    for (const auto& name : names) {
        addObject(name);
    }
}

bool FakeTransformer::calcTransformStart(glm::vec3 ori, glm::vec3 dir, glm::vec3 front) {
    auto x_axis = m_localX;
    auto y_axis = m_localY;
    auto z_axis = glm::cross(m_localX, m_localY);
    std::optional<glm::vec3> t;
    if (m_operation == TransOpt::TRANSLATE) {
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
    else if (m_operation == TransOpt::ROTATE) {
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
    m_operation_mode = m_handler->handleClick(ray_ori, ray_dir);
    if (!calcTransformStart(ray_ori, ray_dir, front)) return false;
    return m_operation_mode != zenovis::INTERACT_NONE;
}

bool FakeTransformer::hoveredAnyHandler(QVector3D ori, QVector3D dir, glm::vec3 front)
{
    if (!m_handler) return false;
    auto ray_ori = QVec3ToGLMVec3(ori);
    auto ray_dir = QVec3ToGLMVec3(dir);
    int mode = m_handler->handleHover(ray_ori, ray_dir);
    if (!calcTransformStart(ray_ori, ray_dir, front)) return false;
    return mode != zenovis::INTERACT_NONE;
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

    auto localZ = glm::cross(m_localX, m_localY);
    auto cur_to_world = glm::mat3(1);
    cur_to_world[0] = m_localX;
    cur_to_world[1] = m_localY;
    cur_to_world[2] = localZ;
    auto cur_to_local = glm::inverse(cur_to_world);

    auto localZOrg = glm::cross(m_localXOrg, m_localYOrg);
    auto pivot_to_world = glm::mat3(1);
    pivot_to_world[0] = m_localXOrg;
    pivot_to_world[1] = m_localYOrg;
    pivot_to_world[2] = localZOrg;
    auto pivot_to_local = glm::inverse(pivot_to_world);

    if (m_operation == TransOpt::TRANSLATE) {
        if (m_operation_mode == zenovis::INTERACT_X) {
            auto cur_pos = hitOnPlane(ori, dir, localZ, m_objects_center);
            if (cur_pos.has_value()) {
                translate(m_trans_start, cur_pos.value(), x_axis, cur_to_local, cur_to_world, pivot_to_local);
            }
        }
        else if (m_operation_mode == zenovis::INTERACT_Y) {
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), y_axis, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::INTERACT_Z) {
            auto cur_pos = hitOnPlane(ori, dir, y_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), z_axis, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::INTERACT_XY) {
            auto cur_pos = hitOnPlane(ori, dir, z_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 1, 0}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::INTERACT_YZ) {
            auto cur_pos = hitOnPlane(ori, dir, x_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {0, 1, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else if (m_operation_mode == zenovis::INTERACT_XZ) {
            auto cur_pos = hitOnPlane(ori, dir, y_axis, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 0, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
        else {
            auto cur_pos = hitOnPlane(ori, dir, front, m_objects_center);
            if (cur_pos.has_value())
                translate(m_trans_start, cur_pos.value(), {1, 1, 1}, cur_to_local, cur_to_world, pivot_to_local);
        }
    }
    else if (m_operation == TransOpt::ROTATE) {
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
    else if (m_operation == TransOpt::SCALE) {
        // make a circle, center is m_objects_center
        // when mouse press, get the circle's radius, r = len(m_objects_center - mouse_start)
        // when mouse move, get the len from center to mouse_pos, d = len(m_objects_center - mouse_pos)
        // so the scale is d / r
        auto t_ctr = vp * glm::vec4(m_objects_center, 1.0f);
        glm::vec2 ctr = t_ctr / t_ctr[3];
        auto len_ctr_start = glm::length(ctr - mouse_start);
        if (len_ctr_start < 0.001) return;

        auto len_ctr_pos = glm::length(ctr - mouse_pos);
        auto scale_size = len_ctr_pos / len_ctr_start;
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

bool FakeTransformer::isTransforming() const {
    return m_status;
}

void FakeTransformer::startTransform() {
    _objects_center_start = m_objects_center;
    _objects_localX_start = m_localX;
    _objects_localY_start = m_localY;
    markObjectsInteractive();
}

void FakeTransformer::createNewTransformNode(NodeLocation& node_location) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto out_sock = node_sync.getPrimSockName(node_location);
    auto new_node_location = node_sync.generateNewNode(node_location,
                                                       "PrimitiveTransform",
                                                       out_sock,
                                                       "prim");

    // make node not visible
    node_sync.updateNodeVisibility(node_location);
    // make new node visible
    node_sync.updateNodeVisibility(new_node_location.value());
}


void FakeTransformer::syncToTransformNode(NodeLocation& node_location,
                                          const std::string& obj_name) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto user_data = m_objects[obj_name]->userData();
    auto translate_data = user_data.getLiterial<zeno::vec3f>("_translate");
    QVector<double> translate = {
        translate_data[0],
        translate_data[1],
        translate_data[2]
    };
    node_sync.updateNodeInputVec(node_location,
                                 "translation",
                                 translate);
    // update scaling
    auto scaling_data = user_data.getLiterial<zeno::vec3f>("_scale");
    QVector<double> scaling = {
        scaling_data[0],
        scaling_data[1],
        scaling_data[2]
    };
    node_sync.updateNodeInputVec(node_location,
                                 "scaling",
                                 scaling);
    // update rotate
    auto rotate_data = user_data.getLiterial<zeno::vec4f>("_rotate");
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
            auto& user_data = obj->userData();

            if (m_operation == TransOpt::TRANSLATE) {
                auto trans = user_data.getLiterial<zeno::vec3f>("_translate");
                trans += other_to_vec<3>(m_trans);
                user_data.setLiterial("_translate", trans);
            }

            if (m_operation == TransOpt::ROTATE) {
                auto rotate = user_data.getLiterial<zeno::vec4f>("_rotate");
                auto pre_q = glm::quat(rotate[3], rotate[0], rotate[1], rotate[2]);
                auto dif_q = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
                auto res_q = glm::toQuat(glm::toMat4(dif_q) * glm::toMat4(pre_q));
                rotate = vec4f(res_q.x, res_q.y, res_q.z, res_q.w);
                user_data.setLiterial("_rotate", rotate);
            }

            if (m_operation == TransOpt::SCALE) {
                auto scale = user_data.getLiterial<zeno::vec3f>("_scale");
                for (int i = 0; i < 3; i++)
                    scale[i] *= m_scale[i];
                user_data.setLiterial("_scale", scale);
            }
        }

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
                if (!linked_transform_node.has_value()) {
                    createNewTransformNode(prim_node_location.value());
                    linked_transform_node = node_sync.checkNodeLinkedSpecificNode(prim_node, "PrimitiveTransform");
                }
                syncToTransformNode(linked_transform_node.value(), obj_name);
            }
        }
    }
    unmarkObjectsInteractive();

    m_trans = {0, 0, 0};
    m_scale = {1, 1, 1};
    m_rotate = {0, 0, 0, 1};

    m_operation_mode = zenovis::INTERACT_NONE;
    m_handler->setMode(zenovis::INTERACT_NONE);
}

void FakeTransformer::toTranslate() {
    if (m_objects.empty()) return;

    auto session = this->session();
    ZASSERT_EXIT(session);

    if (m_operation == TransOpt::TRANSLATE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = TransOpt::TRANSLATE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
    }
    session->set_handler(m_handler);
}

void FakeTransformer::toRotate() {
    if (m_objects.empty()) return;

    auto session = this->session();
    ZASSERT_EXIT(session);

    if (m_operation == TransOpt::ROTATE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = TransOpt::ROTATE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
    }
    session->set_handler(m_handler);
}

void FakeTransformer::toScale() {
    if (m_objects.empty()) return;

    auto session = this->session();
    ZASSERT_EXIT(session);

    if (m_operation == TransOpt::SCALE) {
        m_operation = TransOpt::NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = TransOpt::SCALE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
    }
    session->set_handler(m_handler);
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
    m_status = true;
    for (const auto& [obj_name, obj] : m_objects) {
        markObjectInteractive(obj_name);
    }
}

void FakeTransformer::unmarkObjectsInteractive() {
    m_status = false;
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
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
        break;
    case TransOpt::ROTATE:
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
        break;
    case TransOpt::SCALE:
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(m_objects_center), zeno::other_to_vec<3>(m_localX), zeno::other_to_vec<3>(m_localY), m_handler_scale);
        break;
    case TransOpt::NONE:
        m_handler = nullptr;
    default:
        break;
    }
    session->set_handler(m_handler);
}

void FakeTransformer::changeCoordSys() {
    if (m_coord_sys == zenovis::VIEW_COORD_SYS)
        m_coord_sys = zenovis::WORLD_COORD_SYS;
    else
        ++m_coord_sys;
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

glm::vec3 FakeTransformer::getCenter() const {
    return m_objects_center;
}

void FakeTransformer::clear() {
    m_objects.clear();
    m_trans = {0, 0, 0};
    m_scale = {1, 1, 1};
    m_rotate = {0, 0, 0, 1};
    m_operation = TransOpt::NONE;
    m_handler = nullptr;

    auto session = this->session();
    ZASSERT_EXIT(session);
    session->set_handler(m_handler);
    m_objects_center = {0, 0, 0};
}

void FakeTransformer::translate(glm::vec3 start, glm::vec3 end, glm::vec3 axis, glm::mat3 to_local, glm::mat3 to_world, glm::mat3 org_to_local) {
    auto diff = end - start; // diff in world space
    diff = to_local * diff; // diff in cur local coord
    diff *= axis;
    diff = to_world * diff; // diff in world space
    diff = org_to_local * diff; // diff in pivot local space
    m_trans = diff;
    doTransform();
}

void FakeTransformer::scale(float scale_size, vec3i axis) {
    glm::vec3 scale(1.0f);
    for (int i = 0; i < 3; i++)
        if (axis[i] == 1) scale[i] = std::max(scale_size, 0.1f);
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

void FakeTransformer::doTransform() {
    // qDebug() << "transformer's objects count " << m_objects.size();
    auto lX = m_localXOrg;
    auto lY = m_localYOrg;
    auto lZ = glm::cross(lX, lY);
    auto pivot_to_world = glm::mat4(1);
    pivot_to_world[0] = {lX[0], lX[1], lX[2], 0};
    pivot_to_world[1] = {lY[0], lY[1], lY[2], 0};
    pivot_to_world[2] = {lZ[0], lZ[1], lZ[2], 0};
    pivot_to_world[3] = {m_pivot[0], m_pivot[1], m_pivot[2], 1};
    auto pivot_to_local = glm::inverse(pivot_to_world);

    for (auto &[obj_name, obj] : m_objects) {
        auto& user_data = obj->userData();

        // get transform info
        auto translate = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_translate"));
        auto rotate = zeno::vec_to_other<glm::vec4>(user_data.getLiterial<zeno::vec4f>("_rotate"));
        auto scale = zeno::vec_to_other<glm::vec3>(user_data.getLiterial<zeno::vec3f>("_scale"));
        auto pre_quaternion = glm::quat(rotate[3], rotate[0], rotate[1], rotate[2]);
        auto pre_rotate_matrix = glm::toMat4(pre_quaternion);

        // do this transform
        auto translate_matrix = glm::translate(translate + m_trans);
        auto cur_quaternion = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
        auto rotate_matrix = glm::toMat4(cur_quaternion) * pre_rotate_matrix;
        auto scale_matrix = glm::scale(scale * m_scale);
        auto transform_matrix = pivot_to_world *  translate_matrix *  rotate_matrix * scale_matrix * pivot_to_local;

        if (obj->has_attr("_origin_pos")) {
            // transform pos
            auto &pos = obj->attr<zeno::vec3f>("pos");
            auto &opos = obj->attr<zeno::vec3f>("_origin_pos");
#pragma omp parallel for
            // for (auto &po : pos) {
            for (size_t i = 0; i < pos.size(); ++i) {
                auto p = zeno::vec_to_other<glm::vec3>(opos[i]);
                auto t = transform_matrix * glm::vec4(p, 1.0f);
                auto pt = glm::vec3(t) / t.w;
                pos[i] = zeno::other_to_vec<3>(pt);
            }
        }
        if (obj->has_attr("_origin_nrm")) {
            // transform nrm
            auto &nrm = obj->attr<zeno::vec3f>("nrm");
            auto &onrm = obj->attr<zeno::vec3f>("_origin_nrm");
#pragma omp parallel for
            // for (auto &vec : nrm) {
            for (size_t i = 0; i < nrm.size(); ++i) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                glm::mat3 norm_matrix(transform_matrix);
                norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                auto t = glm::normalize(norm_matrix * n);
                onrm[i] = zeno::other_to_vec<3>(t);
            }
        }
    }
    {
        auto pivot_to_world = glm::mat3(1);
        pivot_to_world[0] = m_localXOrg;
        pivot_to_world[1] = m_localYOrg;
        pivot_to_world[2] = glm::cross(m_localXOrg, m_localYOrg);
        m_objects_center = _objects_center_start + pivot_to_world * m_trans;

        auto cur_quaternion = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
        auto cur_rot = glm::toMat3(cur_quaternion);
        m_localX = cur_rot * _objects_localX_start;
        m_localY = cur_rot * _objects_localY_start;
    }
    m_handler->setCenter(other_to_vec<3>(m_objects_center), other_to_vec<3>(m_localX), other_to_vec<3>(m_localY));
}

}