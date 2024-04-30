#include "transform.h"

#include <zeno/funcs/PrimitiveTools.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Session.h>
#include <zenovis/ObjectsManager.h>
#include "util/uihelper.h"
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include "zassert.h"


namespace zeno {

FakeTransformer::FakeTransformer(ViewportWidget* viewport)
    : m_objects_center(0.0f)
      , m_pivot(0.0f)
      , m_trans(0.0f)
      , m_scale(1.0f)
      , m_rotate({0, 0, 0, 1})
      , m_status(false)
      , m_operation(NONE)
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

void FakeTransformer::addObject(const std::unordered_set<std::string>& names) {
    for (const auto& name : names) {
        addObject(name);
    }
}

void FakeTransformer::removeObject(const std::string& name) {
    if (name.empty()) return;
    auto p = m_objects.find(name);
    if (p == m_objects.end())
        return;
    auto object = p->second;
    if (!object)
        return;

    m_objects_center *= m_objects.size();
    auto& user_data = object->userData();
    zeno::vec3f bmin, bmax;
    if (user_data.has("_bboxMin") && user_data.has("_bboxMax")) {
        bmin = user_data.getLiterial<zeno::vec3f>("_bboxMin");
        bmax = user_data.getLiterial<zeno::vec3f>("_bboxMax");
    } else {
        std::tie(bmin, bmax) = zeno::primBoundingBox(object.get());
        user_data.setLiterial("_bboxMin", bmin);
        user_data.setLiterial("_bboxMax", bmax);
    }
    auto m = zeno::vec_to_other<glm::vec3>(bmax);
    auto n = zeno::vec_to_other<glm::vec3>(bmin);
    m_objects_center -= (m + n) / 2.0f;
    m_objects.erase(p);
    m_objects_center /= m_objects.size();
    m_objectsKeys.erase(name);
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
    if (m_operation == NONE) return;

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

void FakeTransformer::createNewTransformNode(NodeLocation& node_location,
                                             const std::string& obj_name, const std::string& path) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto out_sock = node_sync.getPrimSockName(node_location);
    auto new_node_location = node_sync.generateNewNode(node_location,
                                                       "PrimitiveTransform",
                                                       out_sock,
                                                       "prim");

    auto spObj = m_objects[obj_name];
    auto user_data = spObj->userData();

    auto translate_vec3 = user_data.getLiterial<zeno::vec3f>("_translate");
    QVector<double> translate = {
        translate_vec3[0],
        translate_vec3[1],
        translate_vec3[2]
    };
    node_sync.updateNodeInputVec(new_node_location.value(),
                                 "translation",
                                 translate);

    auto scaling_vec3 = user_data.getLiterial<zeno::vec3f>("_scale");
    QVector<double> scaling = {
        scaling_vec3[0],
        scaling_vec3[1],
        scaling_vec3[2]
    };
    node_sync.updateNodeInputVec(new_node_location.value(),
                                 "scaling",
                                 scaling);

    auto rotate_vec4 = user_data.getLiterial<zeno::vec4f>("_rotate");
    QVector<double> rotate = {
        rotate_vec4[0],
        rotate_vec4[1],
        rotate_vec4[2],
        rotate_vec4[3]
    };
    node_sync.updateNodeInputVec(new_node_location.value(),
                                 "quatRotation",
                                 rotate);

    node_sync.updateNodeInputString(new_node_location.value(), "path", path);

    // make node not visible
    node_sync.updateNodeVisibility(node_location);
    // make new node visible
    node_sync.updateNodeVisibility(new_node_location.value());
}


void FakeTransformer::syncToTransformNode(NodeLocation& node_location,
                                          const std::string& obj_name) {
    auto& node_sync = NodeSyncMgr::GetInstance();

    auto spObj = m_objects[obj_name];
    ZASSERT_EXIT(spObj);
    auto user_data = spObj->userData();
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
            auto spObj = obj;
            ZASSERT_EXIT(spObj);

            auto& user_data = spObj->userData();

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
#if 0
        // sync to node system
        //TODO:
        zeno::scope_exit sp([] {
            IGraphsModel *pGraphs = zenoApp->graphsManager()->currentModel();
            if (pGraphs)
                pGraphs->setApiRunningEnable(true);
        });
        //only update nodes.
        IGraphsModel *pGraphs = zenoApp->graphsManager()->currentModel();
        ZASSERT_EXIT(pGraphs);
        pGraphs->setApiRunningEnable(false);

        std::string primitiveTransformPath;

        for (auto &[obj_name, obj] : m_objects) {
            auto& node_sync = NodeSyncMgr::GetInstance();
            std::optional<NodeLocation> prim_node_location;
            auto spObj = obj;

            if (!spObj->listitemNumberIndex.empty())    //this item comes from a list
            {
                auto namepath = spObj->listitemNameIndex;
                prim_node_location = node_sync.searchNode(namepath.substr(0, namepath.find_first_of("/")));
                primitiveTransformPath += namepath + "(index:" + spObj->listitemNumberIndex + ")" + ";";
            }
            else {                                  //this item is single
                prim_node_location = node_sync.searchNodeOfPrim(obj_name);
            }
            if (!prim_node_location.has_value())
                break;

            auto& prim_node = prim_node_location->node;
            if (node_sync.checkNodeType(prim_node, "PrimitiveTransform") &&
                // prim comes from a exist TransformPrimitive node
                node_sync.checkNodeInputHasValue(prim_node, "translation") &&
                node_sync.checkNodeInputHasValue(prim_node, "quatRotation") &&
                node_sync.checkNodeInputHasValue(prim_node, "scaling")) {
                syncToTransformNode(prim_node_location.value(), obj_name);
            }
            else {
                // prim comes from another type node
                auto linked_transform_node =
                    node_sync.checkNodeLinkedSpecificNode(prim_node_location.value(), "PrimitiveTransform");
                if (linked_transform_node.has_value())
                    // prim links to a exist TransformPrimitive node
                    syncToTransformNode(linked_transform_node.value(), obj_name);
                else
                    // prim doesn't link to a exist TransformPrimitive node
                    createNewTransformNode(prim_node_location.value(), obj_name, primitiveTransformPath);
            }
        }
#endif
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

    if (m_operation == TRANSLATE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = TRANSLATE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
    }
    session->set_handler(m_handler);
}

void FakeTransformer::toRotate() {
    if (m_objects.empty()) return;

    auto session = this->session();
    ZASSERT_EXIT(session);

    if (m_operation == ROTATE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = ROTATE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
    }
    session->set_handler(m_handler);
}

void FakeTransformer::toScale() {
    if (m_objects.empty()) return;

    auto session = this->session();
    ZASSERT_EXIT(session);

    if (m_operation == SCALE) {
        m_operation = NONE;
        m_handler = nullptr;
    }
    else {
        m_operation = SCALE;
        auto scene = session->get_scene();
        ZASSERT_EXIT(scene);
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
    }
    session->set_handler(m_handler);
}

void FakeTransformer::markObjectsInteractive() {
    m_status = true;
    zeno::getSession().objsMan->collect_modify_objs(m_objectsKeys, true);
}

void FakeTransformer::unmarkObjectsInteractive() {
    m_status = false;
    zeno::getSession().objsMan->remove_modify_objs(m_objectsKeys);
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
    if (m_operation == SCALE)
        m_operation = NONE;
    else
        ++m_operation;

    auto session = this->session();
    ZASSERT_EXIT(session);
    auto scene = this->scene();

    switch (m_operation) {
    case TRANSLATE:
        m_handler = zenovis::makeTransHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
        break;
    case ROTATE:
        m_handler = zenovis::makeRotateHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
        break;
    case SCALE:
        m_handler = zenovis::makeScaleHandler(scene, zeno::other_to_vec<3>(m_objects_center), m_handler_scale);
        break;
    case NONE:
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

int FakeTransformer::getTransOpt() {
    return m_operation;
}

void FakeTransformer::setTransOpt(int opt) {
    m_operation = opt;
}

bool FakeTransformer::isTransformMode() const {
    return m_operation != NONE;
}

glm::vec3 FakeTransformer::getCenter() const {
    return m_objects_center;
}

void FakeTransformer::clear() {
    m_objects.clear();
    m_objectsKeys.clear();
    m_trans = {0, 0, 0};
    m_scale = {1, 1, 1};
    m_rotate = {0, 0, 0, 1};
    m_operation = NONE;
    m_handler = nullptr;

    auto session = this->session();
    ZASSERT_EXIT(session);
    session->set_handler(m_handler);
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

void FakeTransformer::startTransform() {
    _objects_center_start = m_objects_center;
    markObjectsInteractive();
}

void FakeTransformer::addObject(const std::string& name) {
    if (name.empty())
        return;

    std::shared_ptr<PrimitiveObject> transformObj;

    auto& objsMan = zeno::getSession().objsMan;

    m_objnodeinfo = objsMan->getObjectAndViewNode(name);
    std::shared_ptr<PrimitiveObject> transObj = std::dynamic_pointer_cast<PrimitiveObject>(m_objnodeinfo.transformingObj);
    if (!transObj) {
        //todo: maybe the transforming obj is a memeber of list object.
        return;
    }

    const std::string& nodecls = m_objnodeinfo.spViewNode->get_nodecls();

    std::shared_ptr<PrimitiveObject> object;
    if (nodecls != "PrimitiveTransform" && nodecls != "TransformPrimitive") {
        object = std::dynamic_pointer_cast<PrimitiveObject>(transObj->clone());
    }
    else {
        object = transObj;
    }

    auto& user_data = object->userData();
    if (!user_data.has("_pivot")) {
        zeno::vec3f bmin, bmax;
        std::tie(bmin, bmax) = zeno::primBoundingBox(object.get());
        zeno::vec3f translate = { 0, 0, 0 };
        user_data.setLiterial("_translate", translate);
        zeno::vec4f rotate = { 0, 0, 0, 1 };
        user_data.setLiterial("_rotate", rotate);
        zeno::vec3f scale = { 1, 1, 1 };
        user_data.setLiterial("_scale", scale);
        auto bboxCenter = (bmin + bmax) / 2;
        user_data.set2("_pivot", bboxCenter);
        if (object->has_attr("pos") && !object->has_attr("_origin_pos")) {
            auto& pos = object->attr<zeno::vec3f>("pos");
            object->verts.add_attr<zeno::vec3f>("_origin_pos") = pos;
        }
        if (object->has_attr("nrm") && !object->has_attr("_origin_nrm")) {
            auto& nrm = object->attr<zeno::vec3f>("nrm");
            object->verts.add_attr<zeno::vec3f>("_origin_nrm") = nrm;
        }
    }
    m_pivot = zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_pivot"));
    m_objects_center *= m_objects.size();
    m_objects_center += m_pivot + zeno::vec_to_other<glm::vec3>(user_data.get2<vec3f>("_translate"));
    m_objects[name] = object;
    m_objects_center /= m_objects.size();
    m_objectsKeys.insert(name);
}

void FakeTransformer::doTransform() {
    // qDebug() << "transformer's objects count " << m_objects.size();
    ZASSERT_EXIT(!m_objects.empty());

    for (auto &[obj_name, obj] : m_objects) {
        auto spObj = obj;
        ZASSERT_EXIT(spObj);
        auto& user_data = spObj->userData();
        user_data.del("_bboxMin");
        user_data.del("_bboxMax");

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
        auto transform_matrix = glm::translate(m_pivot) *  translate_matrix *  rotate_matrix * scale_matrix * glm::translate(-m_pivot);

        if (spObj->has_attr("_origin_pos")) {
            // transform pos
            auto &pos = spObj->attr<zeno::vec3f>("pos");
            auto &opos = spObj->attr<zeno::vec3f>("_origin_pos");
#pragma omp parallel for
            // for (auto &po : pos) {
            for (auto i = 0; i < pos.size(); ++i) {
                auto p = zeno::vec_to_other<glm::vec3>(opos[i]);
                auto t = transform_matrix * glm::vec4(p, 1.0f);
                auto pt = glm::vec3(t) / t.w;
                pos[i] = zeno::other_to_vec<3>(pt);
            }
        }
        if (spObj->has_attr("_origin_nrm")) {
            // transform nrm
            auto &nrm = spObj->attr<zeno::vec3f>("nrm");
            auto &onrm = spObj->attr<zeno::vec3f>("_origin_nrm");
#pragma omp parallel for
            // for (auto &vec : nrm) {
            for (auto i = 0; i < nrm.size(); ++i) {
                auto n = zeno::vec_to_other<glm::vec3>(nrm[i]);
                glm::mat3 norm_matrix(transform_matrix);
                norm_matrix = glm::transpose(glm::inverse(norm_matrix));
                auto t = glm::normalize(norm_matrix * n);
                onrm[i] = zeno::other_to_vec<3>(t);
            }
        }
    }
    m_objects_center = _objects_center_start + m_trans;
    m_handler->setCenter({m_objects_center[0], m_objects_center[1], m_objects_center[2]});

    //TODO: list case
    auto spObj = m_objects.begin()->second;

    scope_exit scope([]() {
        zeno::getSession().setDisableRunning(false);
    });
    zeno::getSession().setDisableRunning(true);

    std::shared_ptr<INode> transNode = m_objnodeinfo.spViewNode;
    if (!transNode) {
        std::shared_ptr<INode> spOriNode = m_objnodeinfo.spViewNode;
        std::shared_ptr<Graph> spGraph = spOriNode->getThisGraph();
        ZASSERT_EXIT(spGraph);
        transNode = spGraph->createNode("PrimitiveTransform");
        ZASSERT_EXIT(transNode);

        spObj->update_key(transNode->get_uuid());

        //把连线关系,view等设置更改。
        EdgeInfo edge;
        edge.outNode = spOriNode->get_name();
        edge.outParam = spOriNode->get_viewobject_output_param();
        edge.inNode = transNode->get_name();
        edge.inParam = "prim";
        spGraph->addLink(edge);

        spOriNode->set_view(false);
        transNode->set_view(true);

        zany originalObj = m_objnodeinfo.transformingObj;
        //原始的对象要隐藏
        auto& objectsMan = zeno::getSession().objsMan;
        //只是为了标记一下view隐藏
        objectsMan->remove_rendering_obj(originalObj);

        //把obj设置到新的transform节点的output端。
        std::string outputparam = transNode->get_viewobject_output_param();
        std::shared_ptr<IParam> spParam = transNode->get_output_param(outputparam);
        transNode->set_result(false, outputparam, spObj);
    }

    {
        //1.直接填写transform的信息
        auto& user_data = spObj->userData();
        auto trans = user_data.getLiterial<zeno::vec3f>("_translate");
        trans += other_to_vec<3>(m_trans);

        auto scale = user_data.getLiterial<zeno::vec3f>("_scale");
        for (int i = 0; i < 3; i++)
            scale[i] *= m_scale[i];
        user_data.setLiterial("_scale", scale);

        auto rotate = user_data.getLiterial<zeno::vec4f>("_rotate");
        auto pre_q = glm::quat(rotate[3], rotate[0], rotate[1], rotate[2]);
        auto dif_q = glm::quat(m_rotate[3], m_rotate[0], m_rotate[1], m_rotate[2]);
        auto res_q = glm::toQuat(glm::toMat4(dif_q) * glm::toMat4(pre_q));
        rotate = vec4f(res_q.x, res_q.y, res_q.z, res_q.w);

        transNode->update_param("translation", trans);
        transNode->update_param("scaling", scale);
        transNode->update_param("quatRotation", rotate);

        //2.登记新的obj到
        auto& objectsMan = zeno::getSession().objsMan;
        objectsMan->collectingObject(spObj, transNode, true);

        //3.渲染端更新加载
        auto mainWin = zenoApp->getMainWindow();
        mainWin->justLoadObjects();
    }
}

}