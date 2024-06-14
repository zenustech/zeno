#include <zenovis/RenderEngine.h>
#include "cameracontrol.h"
#include "zenovis.h"
//#include <zenovis/Camera.h>
#include <zenovis/ObjectsManager.h>
#include "zenomainwindow.h"
#include "nodesview/zenographseditor.h"
#include <zeno/types/UserData.h>
#include "settings/zenosettingsmanager.h"
#include "glm/gtx/quaternion.hpp"
#include "zeno/core/Session.h"
#include <cmath>


using std::string;
using std::unordered_set;
using std::unordered_map;

CameraControl::CameraControl(
            Zenovis* pZenovis,
            std::shared_ptr<zeno::FakeTransformer> transformer,
            std::shared_ptr<zeno::Picker> picker,
            QObject* parent)
    : QObject(parent)
    , m_zenovis(pZenovis)
    , m_transformer(transformer)
    , m_picker(picker)
    , m_res(1, 1)
{
    updatePerspective();
}

void CameraControl::setRes(QVector2D res) {
    m_res = res;
}

glm::vec3 CameraControl::getPos() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->getPos();
}
void CameraControl::setPos(glm::vec3 value) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->setPos(value);
}
glm::vec3 CameraControl::getPivot() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->getPivot();
}
void CameraControl::setPivot(glm::vec3 value) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->setPivot(value);
}
glm::quat CameraControl::getRotation() {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_rotation;
}
void CameraControl::setRotation(glm::quat value) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_rotation = value;
}
zeno::vec3f CameraControl::getCenter() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return zeno::other_to_vec<3>(scene->camera->m_pivot);
}
void CameraControl::setCenter(zeno::vec3f center) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_pivot = zeno::vec_to_other<glm::vec3>(center);
}
bool CameraControl::getOrthoMode() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_ortho_mode;
}
void CameraControl::setOrthoMode(bool orthoMode) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_ortho_mode = orthoMode;
}
float CameraControl::getRadius() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->get_radius();
}

float CameraControl::getFOV() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_fov;
}
void CameraControl::setFOV(float fov) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_fov = fov;
}

float CameraControl::getAperture() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_aperture;
}

void CameraControl::setAperture(float aperture) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_aperture = aperture;
}
float CameraControl::getDisPlane() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->focalPlaneDistance;
}
void CameraControl::setDisPlane(float disPlane) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->focalPlaneDistance = disPlane;
}

void CameraControl::fakeMousePressEvent(QMouseEvent *event)
{
    ZASSERT_EXIT(m_zenovis);
    auto scene = m_zenovis->getSession()->get_scene();
    if (event->button() == Qt::MiddleButton) {
        middle_button_pressed = true;
        if (zeno::getSession().userData().get2<bool>("viewport-depth-aware-navigation", true)) {
            m_hit_posWS = scene->renderMan->getEngine()->getClickedPos(event->x(), event->y());
            if (m_hit_posWS.has_value()) {
                scene->camera->setPivot(m_hit_posWS.value());
            }
        }
    }
    auto m_picker = this->m_picker.lock();
    auto m_transformer = this->m_transformer.lock();
    int button = Qt::NoButton;
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    settings.getViewShortCut(ShortCut_MovingView, button);
    settings.getViewShortCut(ShortCut_RotatingView, button);
    bool bTransform = false;
    auto front = scene->camera->get_lodfront();
    auto dir = screenPosToRayWS(event->x() / res().x(), event->y() / res().y());
    if (m_transformer)
    {
        if (event->buttons() & Qt::LeftButton && !scene->selected.empty() && m_transformer->isTransformMode() &&
            m_transformer->clickedAnyHandler(getPos(), dir, front))
        {
            bTransform = true;
        }
    }
    if (!bTransform && (event->buttons() & button)) {
        m_lastMidButtonPos = event->pos();
    } else if (event->buttons() & Qt::LeftButton) {
        m_boundRectStartPos = event->pos();
        // check if clicked a selected object
        if (bTransform)
        {
            m_transformer->startTransform();
        }
    }
}

void CameraControl::lookTo(zenovis::CameraLookToDir dir) {
    ZASSERT_EXIT(m_zenovis);

    switch (dir) {
    case zenovis::CameraLookToDir::front_view:
        break;
    case zenovis::CameraLookToDir::right_view:
        break;
    case zenovis::CameraLookToDir::top_view:
        break;
    case zenovis::CameraLookToDir::back_view:
        break;
    case zenovis::CameraLookToDir::left_view:
        break;
    case zenovis::CameraLookToDir::bottom_view:
        break;
    case zenovis::CameraLookToDir::back_to_origin:
        break;
    default: break;
    }
    setOrthoMode(true);
    updatePerspective();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::clearTransformer() {
    auto m_transformer = this->m_transformer.lock();
    if (!m_transformer)
        return;
    m_transformer->clear();
}

void CameraControl::changeTransformOperation(const QString &node)
{
    auto m_transformer = this->m_transformer.lock();
    if (!m_transformer)
        return;

    auto opt = m_transformer->getTransOpt();
    m_transformer->clear();

    ZASSERT_EXIT(m_zenovis);

    auto scene = m_zenovis->getSession()->get_scene();
    for (auto const &[key, _] : scene->objectsMan->pairs()) {
        if (key.find(node.toStdString()) != std::string::npos) {
            scene->selected.insert(key);
            m_transformer->addObject(key);
        }
    }
    m_transformer->setTransOpt(opt);
    m_transformer->changeTransOpt();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformOperation(int mode)
{
    auto m_transformer = this->m_transformer.lock();
    if (!m_transformer)
        return;

    switch (mode) {
    case 0: m_transformer->toTranslate(); break;
    case 1: m_transformer->toRotate(); break;
    case 2: m_transformer->toScale(); break;
    default: break;
    }
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformCoordSys()
{
    auto m_transformer = this->m_transformer.lock();
    if (!m_transformer)
        return;
    m_transformer->changeCoordSys();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::resizeTransformHandler(int dir)
{
    auto m_transformer = this->m_transformer.lock();
    if (!m_transformer)
        return;
    m_transformer->resizeHandler(dir);
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::fakeMouseMoveEvent(QMouseEvent *event)
{
    auto m_transformer = this->m_transformer.lock();
    bool ctrl_pressed = event->modifiers() & Qt::ControlModifier;
    bool alt_pressed = event->modifiers() & Qt::AltModifier;

    auto session = m_zenovis->getSession();
    auto scene = session->get_scene();
    float xpos = event->x(), ypos = event->y();

    int moveButton = Qt::NoButton;
    int rotateButton = Qt::NoButton;
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    int moveKey = settings.getViewShortCut(ShortCut_MovingView, moveButton);
    int rotateKey = settings.getViewShortCut(ShortCut_RotatingView, rotateButton);

    bool bTransform = false;
    if (m_transformer) {
        bTransform = m_transformer->isTransforming();
        // check if hover a handler
        auto front = scene->camera->get_lodfront();
        auto dir = screenPosToRayWS(event->x() / res().x(), event->y() / res().y());
        if (!scene->selected.empty() && !(event->buttons() & Qt::LeftButton)) {
            m_transformer->hoveredAnyHandler(getPos(), dir, front);
        }
    }

    if (!bTransform && ctrl_pressed && (event->buttons() & Qt::MiddleButton)) {
        // zoom
        if (zeno::getSession().userData().get2<bool>("viewport-FPN-navigation", false) == false) {
            float dy = ypos - m_lastMidButtonPos.y();
            auto step = 0.99f;
            float scale = glm::pow(step, -dy);
            auto pos = getPos();
            auto pivot = getPivot();
            auto new_pos = (pos - pivot) * scale + pivot;
            setPos(new_pos);
        }
        m_lastMidButtonPos = QPointF(xpos, ypos);
    }
    else if (!bTransform && alt_pressed && (event->buttons() & Qt::MiddleButton)) {
        // rot roll
        float step = 1.0f;
        float ratio = QApplication::desktop()->devicePixelRatio();
        float dy = ypos - m_lastMidButtonPos.y();
        dy *= ratio / m_res[1] * step;
        {
            auto rot = getRotation();
            rot = rot * glm::angleAxis(dy, glm::vec3(0, 0, 1));
            setRotation(rot);
        }
        m_lastMidButtonPos = QPointF(xpos, ypos);
    }
    else if (!bTransform && (event->buttons() & (rotateButton | moveButton))) {
        float ratio = QApplication::desktop()->devicePixelRatio();
        float dx = xpos - m_lastMidButtonPos.x(), dy = ypos - m_lastMidButtonPos.y();
        dx *= ratio / m_res[0];
        dy *= ratio / m_res[1];
        //bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
        Qt::KeyboardModifiers modifiers = event->modifiers();
        if ((moveKey == modifiers) && (event->buttons() & moveButton)) {
            // translate
            if (m_hit_posWS.has_value()) {
                auto ray = screenPosToRayWS(event->x() / res().x(), event->y() / res().y());
                auto new_pos = intersectRayPlane(m_hit_posWS.value(), ray * (-1.0f), getPos(), getViewDir());
                if (new_pos.has_value()) {
                    auto diff = new_pos.value() - getPos();
                    setPivot(getPivot() + diff);
                    setPos(new_pos.value());
                }
            }
            else {
                auto left = getRightDir() * -1.0f;
                auto up = getUpDir();
                auto delta = left * dx + up * dy;
                if (getOrthoMode()) {
                    delta = (left * dx * float(m_res[0]) / float(m_res[1]) + up * dy) * 2.0f;
                }
                auto diff = delta * getRadius();
                setPivot(getPivot() + diff);
                auto new_pos = getPos() + diff;
                setPos(new_pos);
            }
        } else if ((rotateKey == modifiers) && (event->buttons() & rotateButton)) {
            float step = 4.0f;
            dx *= step;
            dy *= step;
            // rot yaw pitch
            setOrthoMode(false);
            {
                auto rot = getRotation();
                auto beforeMat = glm::toMat3(rot);
                rot = glm::angleAxis(-dx, glm::vec3(0, 1, 0)) * rot;
                rot = rot * glm::angleAxis(-dy, glm::vec3(1, 0, 0));
                setRotation(rot);
                auto afterMat = glm::toMat3(rot);
                if (zeno::getSession().userData().get2<bool>("viewport-FPN-navigation", false)) {
                    setPivot(getPos());
                }
                else {
                    auto pos = getPos();
                    auto pivot = getPivot();
                    auto new_pos = afterMat * glm::inverse(beforeMat) * (pos - pivot) + pivot;
                    setPos(new_pos);
                }
            }
        }
        m_lastMidButtonPos = QPointF(xpos, ypos);
    } else if (event->buttons() & Qt::LeftButton) {
        if (m_transformer)
        {
            if (m_transformer->isTransforming()) {
                auto dir = screenPosToRayWS(event->pos().x() / res().x(), event->pos().y() / res().y());

                // mouse pos
                auto mouse_pos = glm::vec2(xpos, ypos);
                mouse_pos[0] = (2 * mouse_pos[0] / res().x()) - 1;
                mouse_pos[1] = 1 - (2 * mouse_pos[1] / res().y());
                // mouse start
                auto mouse_start = glm::vec2(m_boundRectStartPos.x(), m_boundRectStartPos.y());
                mouse_start[0] = (2 * mouse_start[0] / res().x()) - 1;
                mouse_start[1] = 1 - (2 * mouse_start[1] / res().y());

                auto vp = scene->camera->m_proj * scene->camera->m_view;
                m_transformer->transform(getPos(), dir, mouse_start, mouse_pos, scene->camera->get_lodfront(), vp);
                zenoApp->getMainWindow()->updateViewport();
            } else {
                float min_x = std::min((float)m_boundRectStartPos.x(), (float)event->x()) / m_res.x();
                float max_x = std::max((float)m_boundRectStartPos.x(), (float)event->x()) / m_res.x();
                float min_y = std::min((float)m_boundRectStartPos.y(), (float)event->y()) / m_res.y();
                float max_y = std::max((float)m_boundRectStartPos.y(), (float)event->y()) / m_res.y();
                scene->select_box = zeno::vec4f(min_x, min_y, max_x, max_y);
            }
        }
    }
    updatePerspective();
}

void CameraControl::updatePerspective() {
    auto *session = m_zenovis->getSession();
    if (session == nullptr) {
        return;
    }
    m_zenovis->updatePerspective(m_res);
}

void CameraControl::fakeWheelEvent(QWheelEvent *event) {
    int dy = 0;
    if (event->modifiers() & Qt::AltModifier)
        dy = event->angleDelta().x();
    else
        dy = event->angleDelta().y();
    float scale = (dy >= 0) ? 0.89 : 1 / 0.89;
    bool shift_pressed = (event->modifiers() & Qt::ShiftModifier) && !(event->modifiers() & Qt::ControlModifier);
    bool aperture_pressed = (event->modifiers() & Qt::ControlModifier) && !(event->modifiers() & Qt::ShiftModifier);
    bool focalPlaneDistance_pressed =
        (event->modifiers() & Qt::ControlModifier) && (event->modifiers() & Qt::ShiftModifier);
    float delta = dy > 0 ? 1 : -1;
    int button = Qt::NoButton;
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    int scaleKey = settings.getViewShortCut(ShortCut_ScalingView, button);
    if (shift_pressed) {
        auto& inst = ZenoSettingsManager::GetInstance();
        QVariant varEnableShiftChangeFOV = inst.getValue(zsEnableShiftChangeFOV);
        bool bEnableShiftChangeFOV = varEnableShiftChangeFOV.isValid() ? varEnableShiftChangeFOV.toBool() : true;
        if (bEnableShiftChangeFOV) {
            float temp = getFOV() / scale;
            setFOV(temp < 170 ? temp : 170);
        }

    } else if (aperture_pressed) {
        float temp = getAperture() + delta * 0.1;
        setAperture(temp >= 0 ? temp : 0);

    } else if (focalPlaneDistance_pressed) {
        float temp = getDisPlane() + delta * 0.05;
        setDisPlane(temp >= 0.05 ? temp : 0.05);
    } else if (scaleKey == 0 || event->modifiers() & scaleKey){
        if (zeno::getSession().userData().get2<bool>("viewport-FPN-navigation", false)) {
            auto FPN_move_speed = zeno::getSession().userData().get2<int>("viewport-FPN-move-speed", 0);
            FPN_move_speed += dy > 0? 1: -1;
            zeno::getSession().userData().set2("viewport-FPN-move-speed", FPN_move_speed);
            auto pMainWindow = zenoApp->getMainWindow();
            if (pMainWindow) {
                pMainWindow->statusbarShowMessage(zeno::format("First Person Navigation: movement speed level: {}", FPN_move_speed), 10000);
            }
        }
        else {
            auto pos = getPos();
            if (zeno::getSession().userData().get2<bool>("viewport-depth-aware-navigation", true)) {
                auto session = m_zenovis->getSession();
                auto scene = session->get_scene();
                auto hit_posWS = scene->renderMan->getEngine()->getClickedPos(event->x(), event->y());
                if (hit_posWS.has_value()) {
                    auto pivot = hit_posWS.value();
                    auto new_pos = (pos - pivot) * scale + pivot;
                    setPos(new_pos);
                }
                else {
                    auto posOnFloorWS = screenHitOnFloorWS(event->x() / res().x(), event->y() / res().y());
                    auto pivot = posOnFloorWS;
                    if (dot((pivot - pos), getViewDir()) > 0) {
                        auto translate = (pivot - pos) * (1 - scale);
                        if (glm::length(translate) < 0.01) {
                            translate = glm::normalize(translate) * 0.01f;
                        }
                        auto new_pos = translate + pos;
                        setPos(new_pos);
                    }
                    else {
                        auto translate = screenPosToRayWS(event->x() / res().x(), event->y() / res().y()) * getPos().y * (1 - scale);
                        if (getPos().y < 0) {
                            translate *= -1;
                        }
                        auto new_pos = translate + pos;
                        setPos(new_pos);
                    }
                }
            }
            else {
                auto pivot = getPivot();
                auto new_pos = (pos - pivot) * scale + pivot;
                setPos(new_pos);
            }
        }
    }
    updatePerspective();

    if (zenoApp->getMainWindow()->lightPanel != nullptr) {
        zenoApp->getMainWindow()->lightPanel->camApertureEdit->setText(QString::number(getAperture()));
        zenoApp->getMainWindow()->lightPanel->camDisPlaneEdit->setText(QString::number(getDisPlane()));
    }
}

void CameraControl::fakeMouseDoubleClickEvent(QMouseEvent *event)
{
    auto pos = event->pos();
    auto m_picker = this->m_picker.lock();
    if (!m_picker)
        return;
    auto scene = m_zenovis->getSession()->get_scene();
    auto picked_prim = m_picker->just_pick_prim(pos.x(), pos.y());
    if (!picked_prim.empty()) {
        auto primList = scene->objectsMan->pairs();
        QString mtlid;
        for (auto const &[key, ptr]: primList) {
            if (picked_prim == key) {
                auto &ud = ptr->userData();
                mtlid = QString::fromStdString(ud.get2<std::string>("mtlid", ""));
                std::cout<<"selected MatId: "<<ud.get2<std::string>("mtlid", "Default")<<"\n";
            }
        }

        QString subgraph_name;
        QString obj_node_name;
        int type = ZenoSettingsManager::GetInstance().getValue(zsSubgraphType).toInt();
        if (type == SUBGRAPH_TYPE::SUBGRAPH_METERIAL && !mtlid.isEmpty())
        {
            auto graphsMgm = zenoApp->graphsManagment();
            IGraphsModel* pModel = graphsMgm->currentModel();

            const auto& lst = pModel->subgraphsIndice(SUBGRAPH_METERIAL);
            for (const auto& index : lst)
            {
                if (index.data(ROLE_MTLID).toString() == mtlid)
                    subgraph_name = index.data(ROLE_OBJNAME).toString();
            }
        }
        if (subgraph_name.isEmpty())
        {
            auto obj_node_location = zeno::NodeSyncMgr::GetInstance().searchNodeOfPrim(picked_prim);
            subgraph_name = obj_node_location->subgraph.data(ROLE_OBJNAME).toString();
            obj_node_name = obj_node_location->node.data(ROLE_OBJID).toString();
        }

        ZenoMainWindow *pWin = zenoApp->getMainWindow();
        if (pWin) {
            ZenoGraphsEditor *pEditor = pWin->getAnyEditor();
            if (pEditor)
                pEditor->activateTab(subgraph_name, "", obj_node_name);
        }
    }
}

void CameraControl::focus(QVector3D center, float radius) {
    setPivot({float(center.x()), float(center.y()), float(center.z())});
    if (getFOV() >= 1e-6)
        radius /= (getFOV() / 45.0f);
    auto dir = getRotation() * glm::vec3(0, 0, 1) * radius;
    setPos(getPivot() + dir);
    updatePerspective();
}

QVector3D CameraControl::realPos() const {
    auto p = getPos();
    return {p[0], p[1], p[2]};
}

// 计算射线与平面的交点
std::optional<glm::vec3> CameraControl::intersectRayPlane(
        glm::vec3 ray_origin
        , glm::vec3 ray_direction
        , glm::vec3 plane_point
        , glm::vec3 plane_normal
) {
    // 计算射线方向和平面法向量的点积
    float denominator = glm::dot(plane_normal, ray_direction);

    // 如果点积接近于0，说明射线与平面平行或在平面内
    if (glm::abs(denominator) < 1e-6f) {
        return std::nullopt; // 返回空，表示没有交点
    }

    // 计算射线起点到平面上一点的向量
    glm::vec3 diff = plane_point - ray_origin;

    // 计算t值
    float t = glm::dot(diff, plane_normal) / denominator;

    // 如果t < 0，说明交点在射线起点之前，返回空

    if (t < 0) {
        return std::nullopt;
    }

    // 计算交点
    glm::vec3 intersection = ray_origin + t * ray_direction;

    return intersection;
}

// x, y from [0, 1]
glm::vec3 CameraControl::screenPosToRayWS(float x, float y)  {
    x = (x - 0.5) * 2;
    y = (y - 0.5) * (-2);
    float v = std::tan(glm::radians(getFOV()) * 0.5f);
    float aspect = res().x() / res().y();
    auto dir = glm::normalize(glm::vec3(v * x * aspect, v * y, -1));
    return getRotation() * dir;
}

glm::vec3 CameraControl::screenHitOnFloorWS(float x, float y) {
    auto dir = screenPosToRayWS(x, y);
    auto pos = getPos();
    float t = (0 - pos.y) / dir.y;
    return pos + dir * t;
}

void CameraControl::fakeMouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::MiddleButton) {
        middle_button_pressed = false;
    }
    if (event->button() == Qt::LeftButton) {
        auto m_transformer = this->m_transformer.lock();
        auto m_picker = this->m_picker.lock();
        //if (Zenovis::GetInstance().m_bAddPoint == true) {
        //float x = (float)event->x() / m_res.x();
        //float y = (float)event->y() / m_res.y();
        //auto rdir = screenToWorldRay(x, y);
        //auto pos = realPos();
        //float t = (0 - pos.y()) / rdir.y();
        //auto p = pos + rdir * t;

        //float cos_t = std::cos(m_theta);
        //float sin_t = std::sin(m_theta);
        //float cos_p = std::cos(m_phi);
        //float sin_p = std::sin(m_phi);
        //QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        //QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
        //QVector3D right = QVector3D::crossProduct(up, back).normalized();
        //up = QVector3D::crossProduct(right, back).normalized();
        //QVector3D delta = right * x + up * y;

        //zeno::log_info("create point at x={} y={}", p[0], p[1]);

        ////createPointNode(QPointF(p[0], p[1]));

        //Zenovis::GetInstance().m_bAddPoint = false;
        //}

        auto scene = m_zenovis->getSession()->get_scene();
        if (!m_picker || !m_transformer)
            return;

        if (m_transformer->isTransforming()) {
            bool moved = false;
            if (m_boundRectStartPos != event->pos()) {
                // create/modify transform primitive node
                moved = true;
            }
            m_transformer->endTransform(moved);
        } else {
            auto cam_pos = realPos();

            scene->select_box = std::nullopt;
            bool ctrl_pressed = event->modifiers() & Qt::ControlModifier;
            bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
            if (!shift_pressed) {
                scene->selected.clear();
                m_transformer->clear();
            }

            auto onPrimSelected = [this]() {
                auto scene = m_zenovis->getSession()->get_scene();
                ZenoMainWindow *mainWin = zenoApp->getMainWindow();
                mainWin->onPrimitiveSelected(scene->selected);
//                auto pos = event->pos();
//                if (!m_picker)
//                    return;
//
//                auto picked_prim = m_picker->just_pick_prim(pos.x(), pos.y());
//                if (!picked_prim.empty()) {
//                    auto obj_node_location = zeno::NodeSyncMgr::GetInstance().searchNodeOfPrim(picked_prim);
//                    auto subgraph_name = obj_node_location->subgraph.data(ROLE_OBJNAME).toString();
//                    auto obj_node_name = obj_node_location->node.data(ROLE_OBJID).toString();
//                    ZenoMainWindow *pWin = zenoApp->getMainWindow();
//                    if (pWin) {
//                        ZenoGraphsEditor *pEditor = pWin->getAnyEditor();
//                        if (pEditor)
//                            pEditor->activateTab(subgraph_name, "", obj_node_name);
//                    }
//                }
            };

            QPoint releasePos = event->pos();
            if (m_boundRectStartPos == releasePos) {
                if (m_picker->is_draw_mode()) {
                    // zeno::log_info("res_w: {}, res_h: {}", res()[0], res()[1]);
                    m_picker->pick_depth(releasePos.x(), releasePos.y());
                } else {
                    m_picker->pick(releasePos.x(), releasePos.y());
                    m_picker->sync_to_scene();
                    if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_OBJECT)
                        onPrimSelected();
                    m_transformer->clear();
                    m_transformer->addObject(m_picker->get_picked_prims());
                }
                for(auto prim:m_picker->get_picked_prims())
                {
                    if (!prim.empty()) {
                        auto primList = scene->objectsMan->pairs();
                        for (auto const &[key, ptr]: primList) {
                            if (prim == key) {
                                auto &ud = ptr->userData();
                                std::cout<<"selected MatId: "<<ud.get2<std::string>("mtlid", "Default")<<"\n";
                            }
                        }
                    }
                }
            } else {
                int x0 = m_boundRectStartPos.x();
                int y0 = m_boundRectStartPos.y();
                int x1 = releasePos.x();
                int y1 = releasePos.y();
                zeno::SELECTION_MODE mode = zeno::SELECTION_MODE::NORMAL;
                if (shift_pressed == false && ctrl_pressed == false) {
                    mode = zeno::SELECTION_MODE::NORMAL;
                }
                else if (shift_pressed == true && ctrl_pressed == false) {
                    mode = zeno::SELECTION_MODE::APPEND;
                }
                else if (shift_pressed == false && ctrl_pressed == true) {
                    mode = zeno::SELECTION_MODE::REMOVE;
                }

                m_picker->pick(x0, y0, x1, y1, mode);
                m_picker->sync_to_scene();
                if (scene->get_select_mode() == zenovis::PICK_MODE::PICK_OBJECT)
                    onPrimSelected();
                m_transformer->clear();
                m_transformer->addObject(m_picker->get_picked_prims());
                std::cout<<"selected items:"<<m_picker->get_picked_prims().size()<<"\n";
                std::vector<QString> nodes;
                QString sgname;
                for(auto prim:m_picker->get_picked_prims())
                {
                    if (!prim.empty()) {
                        auto primList = scene->objectsMan->pairs();
                        for (auto const &[key, ptr]: primList) {
                            if (prim == key) {
                                auto &ud = ptr->userData();
                                std::cout<<"selected MatId: "<<ud.get2<std::string>("mtlid", "Default")<<"\n";
                            }
                        }
                        auto obj_node_location = zeno::NodeSyncMgr::GetInstance().searchNodeOfPrim(prim);
                        if (!obj_node_location)
                        {
                            return;
                        }
                        auto subgraph_name = obj_node_location->subgraph.data(ROLE_OBJNAME).toString();
                        auto obj_node_name = obj_node_location->node.data(ROLE_OBJID).toString();
                        nodes.push_back(obj_node_name);
//                        ZenoMainWindow *pWin = zenoApp->getMainWindow();
//                        if (pWin) {
//                            ZenoGraphsEditor *pEditor = pWin->getAnyEditor();
//                            if (pEditor)
//                                pEditor->selectTab(subgraph_name, "", obj_node_name);
//                        }
                        sgname = subgraph_name;
                    }
                }

                ZenoMainWindow *pWin = zenoApp->getMainWindow();
                if (pWin) {
                    ZenoGraphsEditor *pEditor = pWin->getAnyEditor();
                    if (pEditor)
                        pEditor->selectTab(sgname, "", nodes);
                }


            }
        }
    }
}

bool CameraControl::fakeKeyPressEvent(int uKey) {
    // viewport focus prim
    if ((uKey & 0xff) == Qt::Key_F && uKey & Qt::AltModifier) {
        auto *scene = m_zenovis->getSession()->get_scene();
        if (scene->selected.size() == 1) {
            std::string nodeId = *scene->selected.begin();
            nodeId = nodeId.substr(0, nodeId.find_first_of(':'));
            zeno::vec3f center;
            float radius;
            if (m_zenovis->getSession()->focus_on_node(nodeId, center, radius)) {
                focus(QVector3D(center[0], center[1], center[2]), radius * 3.0f);
            }
        }
        updatePerspective();
        return true;
    }
    if (!middle_button_pressed) {
        return false;
    }
    float step = glm::pow(1.2f, float(zeno::getSession().userData().get2<int>("viewport-FPN-move-speed", 0)));

    bool processed = false;
    if (uKey == Qt::Key_Q) {
        setPos(getPos() - getUpDir() * step);
        processed = true;
    }
    else if (uKey == Qt::Key_E) {
        setPos(getPos() + getUpDir() * step);
        processed = true;
    }
    else if (uKey == Qt::Key_W) {
        setPos(getPos() + getViewDir() * step);
        processed = true;
    }
    else if (uKey == Qt::Key_S) {
        setPos(getPos() - getViewDir() * step);
        processed = true;
    }
    else if (uKey == Qt::Key_A) {
        setPos(getPos() - getRightDir() * step);
        processed = true;
    }
    else if (uKey == Qt::Key_D) {
        setPos(getPos() + getRightDir() * step);
        processed = true;
    }
    if (processed) {
        updatePerspective();
    }
    return processed;
}

bool CameraControl::fakeKeyReleaseEvent(int uKey) {
    return false;
}
//void CameraControl::createPointNode(QPointF pnt) {
//auto pModel = zenoApp->graphsManagment()->currentModel();
//ZASSERT_EXIT(pModel);
////todo luzh: select specific subgraph to add node.
//const QModelIndex &subgIdx = pModel->index("main");
//NODE_DATA tmpNodeInfo = NodesMgr::createPointNode(pModel, subgIdx, "CreatePoint", {10, 10}, pnt);

//STATUS_UPDATE_INFO info;
//info.role = ROLE_OPTIONS;
//info.newValue = OPT_VIEW;
//pModel->updateNodeStatus(tmpNodeInfo[ROLE_OBJID].toString(), info, subgIdx, true);
//}