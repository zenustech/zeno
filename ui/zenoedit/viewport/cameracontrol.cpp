#include <zenovis/RenderEngine.h>
#include "cameracontrol.h"
#include "zenovis.h"
//#include <zenovis/Camera.h>
#include <zenovis/ObjectsManager.h>
#include "zenomainwindow.h"
#include "nodesview/zenographseditor.h"
#include <zeno/types/UserData.h>
#include "settings/zenosettingsmanager.h"
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

float CameraControl::getRoll() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_roll;
}
void CameraControl::setRoll(float roll) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_roll = roll;
}

float CameraControl::getTheta() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_theta;
}
void CameraControl::setTheta(float theta) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_theta = theta;
}
float CameraControl::getPhi() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_phi;
}
void CameraControl::setPhi(float phi) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_phi = phi;
}
zeno::vec3f CameraControl::getCenter() const {
    auto *scene = m_zenovis->getSession()->get_scene();
    return scene->camera->m_center;
}
void CameraControl::setCenter(zeno::vec3f center) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_center = center;
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
    return scene->camera->m_radius;
}
void CameraControl::setRadius(float radius) {
    auto *scene = m_zenovis->getSession()->get_scene();
    scene->camera->m_radius = radius;
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
    }
    if (scene->camera->m_need_sync) {
        scene->camera->m_need_sync = false;
        if (bool(m_picker) && scene->camera->m_auto_radius) {
            this->m_picker->set_picked_depth_callback([&] (float depth, int x, int y) {
                if (depth < 0.001f) {
                    return;
                }
                glm::vec4 ndc = {0, 0, depth, 1};
                glm::vec4 posCS = glm::inverse(scene->camera->m_proj) * ndc;
                glm::vec4 posVS = posCS / posCS.w;
                glm::vec4 pWS = glm::inverse(scene->camera->m_view) * posVS;
                glm::vec3 p3WS = glm::vec3(pWS.x, pWS.y, pWS.z);
                setRadius(glm::length(scene->camera->m_lodcenter - p3WS));
                setCenter({p3WS.x, p3WS.y, p3WS.z});
            });
            int mid_x = int(this->res().x() * 0.5);
            int mid_y = int(this->res().y() * 0.5);
            this->m_picker->pick_depth(mid_x, mid_y);
        }
    }
    int button = Qt::NoButton;
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    settings.getViewShortCut(ShortCut_MovingView, button);
    settings.getViewShortCut(ShortCut_RotatingView, button);
    bool bTransform = false;
    auto front = scene->camera->m_lodfront;
    auto dir = screenToWorldRay(event->x() / res().x(), event->y() / res().y());
    if (m_transformer)
    {
        if (event->buttons() & Qt::LeftButton && !scene->selected.empty() && m_transformer->isTransformMode() &&
            m_transformer->clickedAnyHandler(realPos(), dir, front))
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

void CameraControl::lookTo(int dir) {
    if (dir < 0 || dir > 6)
        return;
    auto x_axis = QVector3D(1, 0, 0);
    auto y_axis = QVector3D(0, 1, 0);
    auto z_axis = QVector3D(0, 0, 1);

    ZASSERT_EXIT(m_zenovis);
    auto c = getCenter();
    QVector3D center = {c[0], c[1], c[2]};
    auto radius = getRadius();

    switch (dir) {
    case 0:
        // front view
        setTheta(0);
        setPhi(0);
        m_zenovis->updateCameraFront(center + z_axis * radius, -z_axis, y_axis);
        break;
    case 1:
        // right view
        setTheta(0);
        setPhi(-glm::pi<float>() / 2);
        m_zenovis->updateCameraFront(center + x_axis * radius, -x_axis, y_axis);
        break;
    case 2:
        // top view
        setTheta(-glm::pi<float>() / 2);
        setPhi(0);
        m_zenovis->updateCameraFront(center + y_axis * radius, -z_axis, y_axis);
        break;
    case 3:
        // back view
        setTheta(0);
        setPhi(glm::pi<float>());
        m_zenovis->updateCameraFront(center - z_axis * radius, z_axis, y_axis);
        break;
    case 4:
        // left view
        setTheta(0);
        setPhi(glm::pi<float>() / 2);
        m_zenovis->updateCameraFront(center - x_axis * radius, x_axis, y_axis);
        break;
    case 5:
        // bottom view
        setTheta(glm::pi<float>() / 2);
        setPhi(0);
        m_zenovis->updateCameraFront(center - y_axis * radius, y_axis, z_axis);
        break;
    case 6:
        // back to origin
        setCenter({0, 0, 0});
        setRadius(5);
        setTheta(0);
        setPhi(0);
        m_zenovis->updateCameraFront(center, -z_axis, y_axis);
    default: break;
    }
    setOrthoMode(true);
    updatePerspective();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::clearTransformer() {
    if (!m_transformer)
        return;
    m_transformer->clear();
}

void CameraControl::changeTransformOperation(const QString &node)
{
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
    if (!m_transformer)
        return;
    m_transformer->changeCoordSys();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::resizeTransformHandler(int dir)
{
    if (!m_transformer)
        return;
    m_transformer->resizeHandler(dir);
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::fakeMouseMoveEvent(QMouseEvent *event)
{
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
        auto front = scene->camera->m_lodfront;
        auto dir = screenToWorldRay(event->x() / res().x(), event->y() / res().y());
        if (!scene->selected.empty() && !(event->buttons() & Qt::LeftButton)) {
            m_transformer->hoveredAnyHandler(realPos(), dir, front);
        }
    }

    if (!bTransform && ctrl_pressed && (event->buttons() & Qt::MiddleButton)) {
        float ratio = QApplication::desktop()->devicePixelRatio();
        float dx = xpos - m_lastMidButtonPos.x(), dy = ypos - m_lastMidButtonPos.y();
        dx *= ratio / m_res[0];
        dy *= ratio / m_res[1];
        float cos_t = cos(getTheta());
        float sin_t = sin(getTheta());
        float cos_p = cos(getPhi());
        float sin_p = sin(getPhi());
        QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
        QVector3D delta = -back * dy;
        auto c = getCenter();
        QVector3D center = {c[0], c[1], c[2]};
        center += delta * getRadius();
        setCenter({float(center.x()), float(center.y()), float(center.z())});
        m_lastMidButtonPos = QPointF(xpos, ypos);
    }
    else if (!bTransform && alt_pressed && (event->buttons() & Qt::MiddleButton)) {
        float ratio = QApplication::desktop()->devicePixelRatio();
        float dy = ypos - m_lastMidButtonPos.y();
        dy *= ratio / m_res[1];
        float roll = getRoll();
        roll += dy;
        setRoll(roll);
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
            float cos_t = cos(getTheta());
            float sin_t = sin(getTheta());
            float cos_p = cos(getPhi());
            float sin_p = sin(getPhi());
            QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
            QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
            QVector3D right = QVector3D::crossProduct(up, back);
            up = QVector3D::crossProduct(back, right);
            right.normalize();
            up.normalize();
            QVector3D delta = right * dx + up * dy;
            auto c = getCenter();
            QVector3D center = {c[0], c[1], c[2]};
            if (getOrthoMode()) {
                delta = (right * dx * m_res[0] / m_res[1] + up * dy) * 2;
            }
            center += delta * getRadius();
            setCenter({float(center.x()), float(center.y()), float(center.z())});
        } else if ((rotateKey == modifiers) && (event->buttons() & rotateButton)) {
            setOrthoMode(false);
            setTheta(getTheta() - dy * M_PI);
            if (int(abs(getTheta()) / M_PI) % 2 == 0) {
                if (glm::fract(abs(getTheta()) / M_PI) < 0.5) {
                    setPhi(getPhi() + dx * M_PI);
                }
                else {
                    setPhi(getPhi() - dx * M_PI);
                }
            }
            else {
                if (glm::fract(abs(getTheta()) / M_PI) < 0.5) {
                    setPhi(getPhi() - dx * M_PI);
                }
                else {
                    setPhi(getPhi() + dx * M_PI);
                }
            }
        }
        m_lastMidButtonPos = QPointF(xpos, ypos);
    } else if (event->buttons() & Qt::LeftButton) {
        if (m_transformer)
        {
            if (m_transformer->isTransforming()) {
                auto dir = screenToWorldRay(event->pos().x() / res().x(), event->pos().y() / res().y());
                auto camera_pos = realPos();

                // mouse pos
                auto mouse_pos = glm::vec2(xpos, ypos);
                mouse_pos[0] = (2 * mouse_pos[0] / res().x()) - 1;
                mouse_pos[1] = 1 - (2 * mouse_pos[1] / res().y());
                // mouse start
                auto mouse_start = glm::vec2(m_boundRectStartPos.x(), m_boundRectStartPos.y());
                mouse_start[0] = (2 * mouse_start[0] / res().x()) - 1;
                mouse_start[1] = 1 - (2 * mouse_start[1] / res().y());

                auto vp = scene->camera->m_proj * scene->camera->m_view;
                m_transformer->transform(camera_pos, dir, mouse_start, mouse_pos, scene->camera->m_lodfront, vp);
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
        setRadius(getRadius() * scale);
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
//void CameraControl::fakeMouseDoubleClickEvent(QMouseEvent* event) {
void CameraControl::setKeyFrame() {
    //todo
}

void CameraControl::focus(QVector3D center, float radius) {
    setCenter({float(center.x()), float(center.y()), float(center.z())});
    if (getFOV() >= 1e-6)
        radius /= (getFOV() / 45.0f);
    setRadius(radius);
    updatePerspective();
}

QVector3D CameraControl::realPos() const {
    float cos_t = cos(getTheta());
    float sin_t = sin(getTheta());
    float cos_p = cos(getPhi());
    float sin_p = sin(getPhi());
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    auto c = getCenter();
    QVector3D center = {c[0], c[1], c[2]};
    return center - back * getRadius();
}

// x, y from [0, 1]
QVector3D CameraControl::screenToWorldRay(float x, float y) const {
    float cos_t = cos(getTheta());
    float sin_t = sin(getTheta());
    float cos_p = cos(getPhi());
    float sin_p = sin(getPhi());
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    QVector3D right = QVector3D::crossProduct(up, back);
    up = QVector3D::crossProduct(back, right);
    right.normalize();
    up.normalize();
    QMatrix4x4 view;
    view.setToIdentity();
    auto c = getCenter();
    QVector3D center = {c[0], c[1], c[2]};
    view.lookAt(realPos(), center, up);
    x = (x - 0.5) * 2;
    y = (y - 0.5) * (-2);
    float v = std::tan(glm::radians(getFOV()) * 0.5f);
    float aspect = res().x() / res().y();
    auto dir = QVector3D(v * x * aspect, v * y, -1);
    dir = dir.normalized();
    dir = view.inverted().mapVector(dir);
    return dir;
}

QVariant CameraControl::hitOnFloor(float x, float y) const {
    auto dir = screenToWorldRay(x, y);
    auto pos = realPos();
    float t = (0 - pos.y()) / dir.y();
    if (t > 0) {
        auto p = pos + dir * t;
        return p;
    } else {
        return {};
    }
}

void CameraControl::fakeMouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::MiddleButton) {
        middle_button_pressed = false;
    }
    if (event->button() == Qt::LeftButton) {

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
                    if (scene->select_mode == zenovis::PICK_MODE::PICK_OBJECT)
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
                if (scene->select_mode == zenovis::PICK_MODE::PICK_OBJECT)
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
    float cos_t = cos(getTheta());
    float sin_t = sin(getTheta());
    float cos_p = cos(getPhi());
    float sin_p = sin(getPhi());
    zeno::vec3f back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    zeno::vec3f up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    zeno::vec3f left = zeno::cross(up, back);
    auto center = getCenter();
    float step = 1.0f;

    bool processed = false;
    if (uKey == Qt::Key_Q) {
        setCenter(center + zeno::vec3f(0, -1, 0) * step);
        processed = true;
    }
    else if (uKey == Qt::Key_E) {
        setCenter(center + zeno::vec3f(0, 1, 0) * step);
        processed = true;
    }
    else if (uKey == Qt::Key_W) {
        setCenter(center + back * step);
        processed = true;
    }
    else if (uKey == Qt::Key_S) {
        setCenter(center - back * step);
        processed = true;
    }
    else if (uKey == Qt::Key_A) {
        setCenter(center + left * step);
        processed = true;
    }
    else if (uKey == Qt::Key_D) {
        setCenter(center - left * step);
        processed = true;
    }
    if (processed) {
        updatePerspective();
        return true;
    }
    else {
        return false;
    }
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