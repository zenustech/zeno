#include <zenovis/RenderEngine.h>
#include "viewportwidget.h"
#include "zenovis.h"
#include "camerakeyframe.h"
#include "timeline/ztimeline.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include "launch/corelaunch.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "dialog/zrecorddlg.h"
#include "dialog/zrecprogressdlg.h"
#include <zeno/utils/log.h>
#include <zenovis/ObjectsManager.h>
#include <zenovis/Scene.h>
#include <zeno/funcs/ObjectGeometryInfo.h>
#include <zeno/types/UserData.h>
#include <viewport/zenovis.h>
#include <util/log.h>
#include <zenoui/style/zenostyle.h>
//#include <zeno/utils/zeno_p.h>
#include <nodesys/nodesmgr.h>
#include <cmath>
#include <algorithm>
#include <optional>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zenoui/util/uihelper.h>
#include "recordvideomgr.h"


static std::optional<float> ray_sphere_intersect(
    zeno::vec3f const &ray_pos,
    zeno::vec3f const &ray_dir,
    zeno::vec3f const &sphere_center,
    float sphere_radius
) {
    auto &p = ray_pos;
    auto &d = ray_dir;
    auto &c = sphere_center;
    auto &r = sphere_radius;
    float t = zeno::dot(c - p, d);
    if (t < 0) {
        return std::nullopt;
    }
    zeno::vec3f dist = t * d + p - c;
    float l = zeno::length(dist);
    if (l > r) {
        return std::nullopt;
    }
    float t_diff = std::sqrt(r * r - l * l);
    float final_t = t - t_diff;
    return final_t;
}


static bool test_in_selected_bounding(
    QVector3D centerWS,
    QVector3D cam_posWS,
    QVector3D left_normWS,
    QVector3D right_normWS,
    QVector3D up_normWS,
    QVector3D down_normWS
) {
    QVector3D dir =  centerWS - cam_posWS;
    dir.normalize();
    bool left_test = QVector3D::dotProduct(dir, left_normWS) > 0;
    bool right_test = QVector3D::dotProduct(dir, right_normWS) > 0;
    bool up_test = QVector3D::dotProduct(dir, up_normWS) > 0;
    bool down_test = QVector3D::dotProduct(dir, down_normWS) > 0;
    return left_test && right_test && up_test && down_test;
}

CameraControl::CameraControl(QWidget* parent)
    : m_mmb_pressed(false)
    , m_theta(0.)
    , m_phi(0.)
    , m_ortho_mode(false)
    , m_fov(45.)
    , m_radius(5.0)
    , m_res(1, 1)
    , m_aperture(0.1f)
    , m_focalPlaneDistance(2.0f)
{
    transformer = std::make_unique<zeno::FakeTransformer>();
    updatePerspective();
}

void CameraControl::setRes(QVector2D res)
{
    m_res = res;
}

void CameraControl::fakeMousePressEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::MiddleButton) {
        m_lastPos = event->pos();
    }
    else if (event->buttons() & Qt::LeftButton) {
        m_boundRectStartPos = event->pos();
        m_lastPos = event->pos();
        // check if clicked a selected object
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        if (!scene->selected.empty() && mouseEnteredRing(event->x(), event->y())) {
            transformer->startTransform();
        }
    }
}

QVector2D CameraControl::qtCoordToGLCoord(int x, int y) {
    auto w = res()[0];
    auto h = res()[1];
    auto mx = x * 1.0f;
    auto my = y * 1.0f;

    mx = (2 * mx / w) - 1;
    my = 1 - (2 * my / h);
    return {mx, my};
}

bool CameraControl::mouseEnteredRing(int x, int y) {
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    auto world_coord = glm::vec4(transformer->getCenter(), 1.0);
    auto screen_coord = scene->camera->m_proj * scene->camera->m_view * world_coord;
    screen_coord /= screen_coord[3];

    auto cx = screen_coord[0];
    auto cy = screen_coord[1];

    auto mp = qtCoordToGLCoord(x, y);
    auto mx = mp.x();
    auto my = mp.y();

    auto ar = res()[0] / res()[1];

    auto dis = sqrt(((cx - mx) * ar) * ((cx - mx) * ar) + (cy - my) * (cy - my));
    // 0.1 is the bigger radius of ring
    return dis < 0.1;
}

void CameraControl::fakeMouseMoveEvent(QMouseEvent* event)
{
    auto session = Zenovis::GetInstance().getSession();
    auto scene = session->get_scene();

    if (event->buttons() & Qt::MiddleButton) {
        float ratio = QApplication::desktop()->devicePixelRatio();
        float xpos = event->x(), ypos = event->y();
        float dx = xpos - m_lastPos.x(), dy = ypos - m_lastPos.y();
        dx *= ratio / m_res[0];
        dy *= ratio / m_res[1];
        bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
        if (shift_pressed)
        {
            float cos_t = cos(m_theta);
            float sin_t = sin(m_theta);
            float cos_p = cos(m_phi);
            float sin_p = sin(m_phi);
            QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
            QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
            QVector3D right = QVector3D::crossProduct(up, back);
            up = QVector3D::crossProduct(back, right);
            right.normalize();
            up.normalize();
            QVector3D delta = right * dx + up * dy;
            m_center += delta * m_radius;
        }
        else
        {
            m_theta -= dy * M_PI;
            m_phi += dx * M_PI;
        }
        m_lastPos = QPointF(xpos, ypos);
    }
    else if (event->buttons() & Qt::LeftButton) {
        if (transformer->isTransforming()) {
            bool alt_pressed = event->modifiers() & Qt::AltModifier;
            bool ctrl_pressed = event->modifiers() & Qt::ControlModifier;
            bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
            glm::mat4 transform_matrix(1.0f);

            auto transform_center = transformer->getCenter();
            QVector3D center_qvec(transform_center[0], transform_center[1], transform_center[2]);
            zeno::vec3f center_zvec(transform_center[0], transform_center[1], transform_center[2]);

            std::set<std::unique_ptr<zenovis::IGraphicDraw>> interactingGraphics;
            QVector3D x_axis(1, 0, 0);
            QVector3D y_axis(0, 1, 0);
            QVector3D z_axis(0, 0, 1);

            QVariant start, end;
            QVector3D start_vec, end_vec;
            // get transform matrix
            if (ctrl_pressed && alt_pressed) {
                // rotate
                if (m_pressedKeys.contains(Qt::Key_Z)) {
                    // rotate along x=center_qvec.x plane
                    start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                       x_axis, center_qvec);
                    end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                     x_axis, center_qvec);
                    if (start.isValid() && end.isValid()) {
                        start_vec = start.value<QVector3D>() - center_qvec;
                        end_vec = end.value<QVector3D>() - center_qvec;
                        transformer->rotate(start_vec, end_vec, x_axis);
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 0, 0}));
                    }
                }
                else if (m_pressedKeys.contains(Qt::Key_X)) {
                    // rotate along y=center_qvec.y plane
                    start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                       y_axis, center_qvec);
                    end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                     y_axis, center_qvec);
                    if (start.isValid() && end.isValid()) {
                        start_vec = start.value<QVector3D>() - center_qvec;
                        end_vec = end.value<QVector3D>() - center_qvec;
                        transformer->rotate(start_vec, end_vec, y_axis);
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 1, 0}));
                    }
                }
                else if (m_pressedKeys.contains(Qt::Key_C)) {
                    // rotate along z=center_qvec.z plane
                    start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                       z_axis, center_qvec);
                    end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                     z_axis, center_qvec);
                    if (start.isValid() && end.isValid()) {
                        start_vec = start.value<QVector3D>() - center_qvec;
                        end_vec = end.value<QVector3D>() - center_qvec;
                        transformer->rotate(start_vec, end_vec, z_axis);
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 0, 1}));
                    }
                }
            }
            else if (alt_pressed) {
                // scale
                auto vp = scene->camera->m_proj * scene->camera->m_view;
                auto screen_center = vp * glm::vec4(transform_center, 1.0f);
                screen_center /= screen_center[3];
                QVector2D screen_center_2d(screen_center[0], screen_center[1]);
                QVector2D mouse_pos = qtCoordToGLCoord(event->x(), event->y());
                float scale_size = mouse_pos.distanceToPoint(screen_center_2d);
                // printf("%.2lf\n", scale_size);

                if (m_pressedKeys.contains(Qt::Key_A)) {
                    // scale along a plane
                    if (m_pressedKeys.contains(Qt::Key_Z)) {
                        // scale along x=center_qvec.x plane
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              y_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{0, 1, 1}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec,{0, 1, 1}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_X)) {
                        // scale along y=center_qvec.y plane
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              y_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{1, 0, 1}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 0, 1}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_C)) {
                        // scale along z=center_qvec.z plane
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              z_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{1, 1, 0}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 1, 0}));
                    }
                }
                else {
                    // scale along an axis
                    if (m_pressedKeys.contains(Qt::Key_Z)) {
                        // scale along x axis
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              y_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{1, 0, 0}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 0, 0}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_X)) {
                        // scale along y=center_qvec.y plane
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              y_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{0, 1, 0}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 1, 0}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_C)) {
                        auto cur = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                              z_axis, center_qvec);
                        if (cur.isValid()) {
                            transformer->scale(cur.value<QVector3D>() - center_qvec,{0, 0, 1}, scale_size);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 0, 1}));
                    }
                }
            }
            else if (ctrl_pressed){
                // translate
                if (m_pressedKeys.contains(Qt::Key_A)) {
                    // translate along a plane
                    if (m_pressedKeys.contains(Qt::Key_Z)) {
                        // translate along x=center_qvec.x plane
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           x_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         x_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),
                                                   end.value<QVector3D>(), {0, 1, 1});
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec,{0, 1, 1}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_X)) {
                        // translate along y=center_qvec.y plane
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           y_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         y_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),
                                                   end.value<QVector3D>(), {1, 0, 1});
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 0, 1}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_C)) {
                        // translate along z=center_qvec.z plane
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           z_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         z_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),
                                                   end.value<QVector3D>(), {1, 1, 0});
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 1, 0}));
                    }
                }
                else {
                    // translate along an axis
                    QVector3D translate_axis;
                    if (m_pressedKeys.contains(Qt::Key_Z)) {
                        // translate along x axis
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           y_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         y_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),end.value<QVector3D>(), x_axis);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {1, 0, 0}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_X)) {
                        // translate along y=center_qvec.y plane
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           z_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         z_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),end.value<QVector3D>(), y_axis);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 1, 0}));
                    }
                    else if (m_pressedKeys.contains(Qt::Key_C)) {
                        // translate along z=center_qvec.z plane
                        start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                           y_axis, center_qvec);
                        end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                         y_axis, center_qvec);
                        if (start.isValid() && end.isValid()) {
                            transformer->translate(start.value<QVector3D>(),end.value<QVector3D>(), z_axis);
                        }
                        interactingGraphics.insert(
                            zenovis::makeGraphicInteractingAxis(scene, center_zvec, {0, 0, 1}));
                    }
                }
            }
            else {
                // free translate (along sight plane)
                QVector3D camera_front(scene->camera->m_lodfront[0],
                                       scene->camera->m_lodfront[1],
                                       scene->camera->m_lodfront[2]);
                start = hitOnPlane((float)m_lastPos.x() / res().x(), (float)m_lastPos.y() / res().y(),
                                   camera_front, center_qvec);
                end = hitOnPlane((float)event->x() / res().x(), (float)event->y() / res().y(),
                                 camera_front, center_qvec);
                if (start.isValid() && end.isValid()) {
                    transformer->translate(start.value<QVector3D>(),end.value<QVector3D>(), {1, 1, 1});
                }
            }
            session->set_interacting_graphics(interactingGraphics);
            m_lastPos = event->pos();
            zenoApp->getMainWindow()->updateViewport();
        }
        else {
            float min_x = std::min((float)m_boundRectStartPos.x(), (float)event->x()) / m_res.x();
            float max_x = std::max((float)m_boundRectStartPos.x(), (float)event->x()) / m_res.x();
            float min_y = std::min((float)m_boundRectStartPos.y(), (float)event->y()) / m_res.y();
            float max_y = std::max((float)m_boundRectStartPos.y(), (float)event->y()) / m_res.y();
            scene->select_box = zeno::vec4f(min_x, min_y, max_x, max_y);
        }
    }
    else {
        if (mouseEnteredRing(event->x(), event->y())) {
            Zenovis::GetInstance().getSession()->set_hovered_graphic("ring");
        }
        else {
            Zenovis::GetInstance().getSession()->set_hovered_graphic("");
        }
    }
    updatePerspective();
}

void CameraControl::updatePerspective()
{
    float cx = m_center[0], cy = m_center[1], cz = m_center[2];
    Zenovis::GetInstance().updatePerspective(m_res, PerspectiveInfo(cx, cy, cz, m_theta, m_phi, m_radius, m_fov, m_ortho_mode, m_aperture, m_focalPlaneDistance));
}

void CameraControl::fakeWheelEvent(QWheelEvent* event)
{
    int dy = event->angleDelta().y();
    float scale = (dy >= 0) ? 0.89 : 1 / 0.89;
    bool shift_pressed = (event->modifiers() & Qt::ShiftModifier) && !(event->modifiers() & Qt::ControlModifier);
    bool aperture_pressed = (event->modifiers() & Qt::ControlModifier) && !(event->modifiers() & Qt::ShiftModifier);
    bool focalPlaneDistance_pressed = (event->modifiers() & Qt::ControlModifier) && (event->modifiers() & Qt::ShiftModifier);
    float delta = dy > 0? 1: -1;
    if (shift_pressed){
        float temp = m_fov / scale;
        m_fov = temp < 170 ? temp : 170;

    }
    else if (aperture_pressed) {
        float temp = m_aperture += delta * 0.01;
        m_aperture = temp >= 0 ? temp : 0;  

    }
    else if (focalPlaneDistance_pressed) {
        float temp = m_focalPlaneDistance + delta*0.05;
        m_focalPlaneDistance = temp >= 0.05 ? temp : 0.05;
    }
    else {
        m_radius *= scale;
    }
    updatePerspective();
}

void CameraControl::setKeyFrame()
{
    //todo
}

void CameraControl::focus(QVector3D center, float radius)
{
    m_center = center;
    if (m_fov >= 1e-6)
        radius /= (m_fov / 45.0f);
    m_radius = radius;
    updatePerspective();
}

QVector3D CameraControl::realPos() const {
    float cos_t = std::cos(m_theta);
    float sin_t = std::sin(m_theta);
    float cos_p = std::cos(m_phi);
    float sin_p = std::sin(m_phi);
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    return m_center - back * m_radius;
}

// x, y from [0, 1]
QVector3D CameraControl::screenToWorldRay(float x, float y) const {
    float cos_t = cos(m_theta);
    float sin_t = sin(m_theta);
    float cos_p = cos(m_phi);
    float sin_p = sin(m_phi);
    QVector3D back(cos_t * sin_p, sin_t, -cos_t * cos_p);
    QVector3D up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    QVector3D right = QVector3D::crossProduct(up, back);
    up = QVector3D::crossProduct(back, right);
    right.normalize();
    up.normalize();
    QMatrix4x4 view;
    view.setToIdentity();
    view.lookAt(realPos(), m_center, up);
    x = (x - 0.5) * 2;
    y = (y - 0.5) * (-2);
    float v = std::tan(m_fov * M_PI / 180.f * 0.5f);
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

QVariant CameraControl::hitOnPlane(float x, float y, QVector3D n, QVector3D p) const {
    auto dir = screenToWorldRay(x, y);
    auto pos = realPos();
    float t = (n.x()*p.x() + n.y()*p.y() + n.z()*p.z() - n.x()*pos.x() - n.y()*pos.y() - n.z()*pos.z()) /
              (n.x()*dir.x() + n.y()*dir.y() + n.z()*dir.z());
    if (t > 0)
        return pos + dir * t;
    else
        return {};
}

void CameraControl::fakeMouseReleaseEvent(QMouseEvent *event) {
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

        auto scene = Zenovis::GetInstance().getSession()->get_scene();

        if (transformer->isTransforming()) {
            if (m_boundRectStartPos != event->pos()) {
                // create/modify transform primitive node
                IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
                for (const auto &obj_name : scene->selected) {
                    QString node_id(obj_name.substr(0, obj_name.find_first_of(':')).c_str());
                    auto search_result = pModel->search(node_id, SEARCH_NODEID);
                    auto subgraph_index = search_result[0].subgIdx;
                    auto node_index = search_result[0].targetIdx;
                    auto inputs = node_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                    if (node_id.contains("TransformPrimitive")  &&
                        inputs["translation"].linkIndice.empty() &&
                        inputs["eulerXYZ"].linkIndice.empty() &&
                        inputs["scaling"].linkIndice.empty()) {
                        transformer->syncToTransformNode(node_id, pModel, node_index, subgraph_index);
                    }
                    else {
                        auto linked_transform_node_index =
                            transformer->linkedToVisibleTransformNode(node_index, pModel).value<QModelIndex>();
                        if (linked_transform_node_index.isValid()) {
                            auto linked_transform_node_id = linked_transform_node_index.data(ROLE_OBJID).toString();
                            transformer->syncToTransformNode(linked_transform_node_id, pModel, linked_transform_node_index, subgraph_index);
                        }
                        else
                            transformer->createNewTransformNode(node_id, pModel, node_index, subgraph_index);
                    }
                }
            }
            transformer->endTransform();
        }
        else {
            auto cam_pos = realPos();

            scene->select_box = std::nullopt;
            bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
            if (!shift_pressed) {
                scene->selected.clear();
                transformer->clear();
            }

            QPoint releasePos = event->pos();
            if (m_boundRectStartPos == releasePos) {
                float x = (float)event->x() / m_res.x();
                float y = (float)event->y() / m_res.y();
                auto rdir = screenToWorldRay(x, y);
                float min_t = std::numeric_limits<float>::max();
                std::string name("");
                for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
                    zeno::vec3f center;
                    float radius;
                    zeno::vec3f ro(cam_pos[0], cam_pos[1], cam_pos[2]);
                    zeno::vec3f rd(rdir[0], rdir[1], rdir[2]);
                    if (zeno::objectGetFocusCenterRadius(ptr, center, radius)) {
                        if (auto ret = ray_sphere_intersect(ro, rd, center, radius)) {
                            float t = *ret;
                            if (t < min_t) {
                                min_t = t;
                                name = key;
                            }
                        }
                    }
                }
                if (!name.empty()) {
                    if (scene->selected.count(name) > 0) {
                        scene->selected.erase(name);
                        transformer->removeObject(name);
                    } else {
                        scene->selected.insert(name);
                        transformer->addObject(name);
                        // printf("selected %s\n", name.c_str());
                    }
                }
            } else {
                float min_x = std::min((float)m_boundRectStartPos.x(), (float)releasePos.x());
                float max_x = std::max((float)m_boundRectStartPos.x(), (float)releasePos.x());
                float min_y = std::min((float)m_boundRectStartPos.y(), (float)releasePos.y());
                float max_y = std::max((float)m_boundRectStartPos.y(), (float)releasePos.y());
                auto left_up = screenToWorldRay(min_x / m_res.x(), min_y / m_res.y());
                auto left_down = screenToWorldRay(min_x / m_res.x(), max_y / m_res.y());
                auto right_up = screenToWorldRay(max_x / m_res.x(), min_y / m_res.y());
                auto right_down = screenToWorldRay(max_x / m_res.x(), max_y / m_res.y());
                auto cam_posWS = realPos();
                auto left_normWS = QVector3D::crossProduct(left_down, left_up);
                auto right_normWS = QVector3D::crossProduct(right_up, right_down);
                auto up_normWS = QVector3D::crossProduct(left_up, right_up);
                auto down_normWS = QVector3D::crossProduct(right_down, left_down);

                std::vector<std::string> passed_prim;
                for (auto const &[key, ptr] : scene->objectsMan->pairs()) {
                    zeno::vec3f c;
                    float radius;
                    if (zeno::objectGetFocusCenterRadius(ptr, c, radius)) {
                        bool passed = test_in_selected_bounding(QVector3D(c[0], c[1], c[2]), cam_pos, left_normWS,
                                                                right_normWS, up_normWS, down_normWS);
                        if (passed) {
                            passed_prim.push_back(key);
                            transformer->addObject(key);
                        }
                    }
                }
                scene->selected.insert(passed_prim.begin(), passed_prim.end());

            }
        }

        if (!scene->selected.empty()) {
            auto center = transformer->getCenter();
            zeno::vec3f ring_center(center[0], center[1], center[2]);
            std::map<std::string, std::unique_ptr<zenovis::IGraphicInteractDraw>> interactGraphics;
            interactGraphics["ring"] = zenovis::makeGraphicRing(scene, ring_center);
            interactGraphics["axis"] = zenovis::makeGraphicInteractAxis(scene, ring_center);
            Zenovis::GetInstance().getSession()->set_interactive_graphics(interactGraphics);

            zenoApp->getMainWindow()->updateViewport();
        }

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        mainWin->onPrimitiveSelected(scene->selected);
    }
}

void CameraControl::addPressedKey(int key) {
    m_pressedKeys += key;
}

void CameraControl::rmvPressedKey(int key) {
    m_pressedKeys -= key;
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

ViewportWidget::ViewportWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_camera(nullptr)
    , updateLightOnce(true)
{
    QSurfaceFormat fmt;
    int nsamples = 16;  // TODO: adjust in a zhouhang-panel
    fmt.setSamples(nsamples);
    fmt.setVersion(3, 2);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    setFormat(fmt);

    m_camera = std::make_shared<CameraControl>();
    Zenovis::GetInstance().m_camera_control = m_camera.get();
}

ViewportWidget::~ViewportWidget()
{
}

namespace {
struct OpenGLProcAddressHelper {
    inline static QOpenGLContext *ctx;

    static void *getProcAddress(const char *name) {
        return (void *)ctx->getProcAddress(name);
    }
};
}

void ViewportWidget::initializeGL()
{
    OpenGLProcAddressHelper::ctx = context();
    Zenovis::GetInstance().loadGLAPI((void *)OpenGLProcAddressHelper::getProcAddress);
    Zenovis::GetInstance().initializeGL();
}

void ViewportWidget::resizeGL(int nx, int ny)
{
    float ratio = devicePixelRatioF();
    zeno::log_trace("nx={}, ny={}, dpr={}", nx, ny, ratio);
    m_camera->setRes(QVector2D(nx * ratio, ny * ratio));
    m_camera->updatePerspective();
}

QVector2D ViewportWidget::cameraRes() const
{
    return m_camera->res();
}

void ViewportWidget::setCameraRes(const QVector2D& res)
{
    m_camera->setRes(res);
}

void ViewportWidget::updatePerspective()
{
    m_camera->updatePerspective();
}

void ViewportWidget::paintGL()
{
    Zenovis::GetInstance().paintGL();
    if(updateLightOnce){
        auto scene = Zenovis::GetInstance().getSession()->get_scene();

        if(scene->objectsMan->lightObjects.size() > 0){
            zenoApp->getMainWindow()->updateLightList();
            updateLightOnce = false;
        }
    }

    if (!record_path.empty() /*&& f <= frame_end*/) //py has bug: frame_end not initialized.
    {
        int f = Zenovis::GetInstance().getCurrentFrameId();
        auto record_file = zeno::format("{}/{:06d}.png", record_path, f);
        checkRecord(record_file, record_res);
    }
}

void ViewportWidget::checkRecord(std::string a_record_file, QVector2D a_record_res)
{
    if (!record_path.empty() /*&& f <= frame_end*/) //py has bug: frame_end not initialized.
    {
        QVector2D oldRes = m_camera->res();
        m_camera->setRes(a_record_res);
        m_camera->updatePerspective();
        auto extname = QFileInfo(QString::fromStdString(a_record_file)).suffix().toStdString();
        Zenovis::GetInstance().getSession()->do_screenshot(a_record_file, extname);
        m_camera->setRes(oldRes);
        m_camera->updatePerspective();
        //if f == self.frame_end:
        //    self.parent_widget.record_video.finish_record()
    }
}

void ViewportWidget::mousePressEvent(QMouseEvent* event)
{
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* event)
{
    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ViewportWidget::wheelEvent(QWheelEvent* event)
{
    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}

void ViewportWidget::mouseReleaseEvent(QMouseEvent *event) {    
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event); 
    update();
}

void ViewportWidget::addPressedKey(int key) {
    m_camera->addPressedKey(key);
}

void ViewportWidget::rmvPressedKey(int key) {
    m_camera->rmvPressedKey(key);
}

/*
QDMDisplayMenu::QDMDisplayMenu()r("Show Grid")
{
    setTitle(tr("Display"));
    QAction* pAction = new QAction(tr("Show Grid"), this);
    pAction->setCheckable(true);
    pAction->setChecked(true);
    addAction(pAction);

    pAction = new QAction(tr("Background Color"), this);
    addAction(pAction);

    addSeparator();

    pAction = new QAction(tr("Smooth Shading"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);
    addAction(pAction);

    pAction = new QAction(tr("Wireframe"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);

    addSeparator();

    pAction = new QAction(tr("Enable PBR"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);
    addAction(pAction);

    pAction = new QAction(tr("Enable GI"), this);
    pAction->setCheckable(true);
    pAction->setChecked(false);
    addAction(pAction);

    addSeparator();

    pAction = new QAction(tr("Camera Keyframe"), this);
    addAction(pAction);

    addSeparator();

    pAction = new QAction(tr("English / Chinese"), this);
    pAction->setCheckable(true);
    pAction->setChecked(true);
    addAction(pAction);
}

QDMRecordMenu::QDMRecordMenu()
{
    setTitle(tr("Record"));

    QAction* pAction = new QAction(tr("Screenshot"), this);
    pAction->setShortcut(QKeySequence("F12"));
    addAction(pAction);

    pAction = new QAction(tr("Record Video"), this);
    pAction->setShortcut(QKeySequence(tr("Shift+F12")));
    addAction(pAction);
}
*/


DisplayWidget::DisplayWidget(ZenoMainWindow* pMainWin)
    : QWidget(pMainWin)
    , m_view(nullptr)
    , m_timeline(nullptr)
    , m_mainWin(pMainWin)
    , m_pTimer(nullptr)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    /*
    ZMenuBar* menuBar = new ZMenuBar;
    menuBar->setMaximumHeight(26);

    QDMDisplayMenu* menuDisplay = new QDMDisplayMenu;
    menuBar->addMenu(menuDisplay);
    QDMRecordMenu* recordDisplay = new QDMRecordMenu;
    menuBar->addMenu(recordDisplay);

    pLayout->addWidget(menuBar);
    */

    m_view = new ViewportWidget;
    m_view->setMouseTracking(true);
    pLayout->addWidget(m_view);

    m_timeline = new ZTimeline;
    pLayout->addWidget(m_timeline);
    setLayout(pLayout);

    //RecordVideoDialog
    m_camera_keyframe = new CameraKeyframeWidget;
    Zenovis::GetInstance().m_camera_keyframe = m_camera_keyframe;

	connect(&Zenovis::GetInstance(), SIGNAL(frameUpdated(int)), m_timeline, SLOT(onTimelineUpdate(int)));
    connect(m_timeline, SIGNAL(playForward(bool)), this, SLOT(onPlayClicked(bool)));
	connect(m_timeline, SIGNAL(sliderValueChanged(int)), this, SLOT(onSliderValueChanged(int)));
	connect(m_timeline, SIGNAL(run()), this, SLOT(onRun()));
    connect(m_timeline, SIGNAL(alwaysChecked()), this, SLOT(onRun()));
    connect(m_timeline, SIGNAL(kill()), this, SLOT(onKill()));

    //connect(m_view, SIGNAL(sig_Draw()), this, SLOT(onRun()));

    auto graphs = zenoApp->graphsManagment();
    connect(&*graphs, SIGNAL(modelDataChanged()), this, SLOT(onModelDataChanged()));

	m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

DisplayWidget::~DisplayWidget()
{

}

void DisplayWidget::init()
{
    //m_camera->installEventFilter(this);
}

TIMELINE_INFO DisplayWidget::timelineInfo()
{
    TIMELINE_INFO info;
    info.bAlways = m_timeline->isAlways();
    info.beginFrame = m_timeline->fromTo().first;
    info.endFrame = m_timeline->fromTo().second;
    return info;
}

void DisplayWidget::setTimelineInfo(TIMELINE_INFO info)
{
    m_timeline->setAlways(info.bAlways);
    m_timeline->setFromTo(info.beginFrame, info.endFrame);
}

QSize DisplayWidget::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(QSize(12, 400));
}

void DisplayWidget::updateFrame(const QString &action) // cihou optix
{
    if (m_mainWin && m_mainWin->inDlgEventLoop())
        return;

    if (action == "newFrame") {
        m_pTimer->stop();
        //zeno::log_warn("stop");
        return;
    } else if (action == "finishFrame") {
        if (isOptxRendering()) {
            m_pTimer->start(m_updateFeq);
        }
    } else if (!action.isEmpty()) {
        if (action == "optx") {
            m_pTimer->start(m_updateFeq);
        } else {
            m_pTimer->stop();
        }
    }
    m_view->update();
}

void DisplayWidget::onFinished()
{
    int frameid_ui = m_timeline->value();
    if (frameid_ui != Zenovis::GetInstance().getCurrentFrameId())
    {
        Zenovis::GetInstance().setCurrentFrameId(frameid_ui);
        updateFrame();
        onPlayClicked(false);
        BlockSignalScope scope(m_timeline);
        m_timeline->setPlayButtonToggle(false);
    }
}

bool DisplayWidget::isOptxRendering() const
{
    auto& inst = Zenovis::GetInstance();
    auto sess = inst.getSession();
    if (!sess)
        return false;
    auto scene = sess->get_scene();
    if (!scene)
        return false;

    return (scene->renderMan && scene->renderMan->getDefaultEngineName() == "optx");
}

void DisplayWidget::onModelDataChanged()
{
    if (m_timeline->isAlways())
    {
        onRun();
    }
}

void DisplayWidget::onPlayClicked(bool bChecked)
{
    if (bChecked)
    {
        m_pTimer->start(m_sliderFeq);
    }
    else
    {
        if (!isOptxRendering())
            m_pTimer->stop();
    }
    Zenovis::GetInstance().startPlay(bChecked);
}

void DisplayWidget::onSliderValueChanged(int frame)
{
    m_mainWin->clearErrorMark();

    if (m_timeline->isAlways())
    {
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel* pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        launchProgram(pModel, frame, frame);
    }
    else
    {
        Zenovis::GetInstance().setCurrentFrameId(frame);
        updateFrame();
        onPlayClicked(false);
        BlockSignalScope scope(m_timeline);
        m_timeline->setPlayButtonToggle(false);
    }
}

void DisplayWidget::onRun()
{
    m_mainWin->clearErrorMark();

    Zenovis::GetInstance().getSession()->get_scene()->selected.clear();

    QPair<int, int> fromTo = m_timeline->fromTo();
    int beginFrame = fromTo.first;
    int endFrame = fromTo.second;
    if (endFrame >= beginFrame && beginFrame >= 0)
    {
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel* pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        launchProgram(pModel, beginFrame, endFrame);
    }
    else
    {

    }

    m_view->updateLightOnce = true;
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::keyPressEvent(QKeyEvent* event) {
    if (!event->isAutoRepeat())
        m_view->addPressedKey(event->key());
}

void DisplayWidget::keyReleaseEvent(QKeyEvent* event) {
    if (!event->isAutoRepeat())
        m_view->rmvPressedKey(event->key());
}


void DisplayWidget::onRecord()
{
    auto& inst = Zenovis::GetInstance();

    auto& ptr = zeno::getSession().globalComm;
    ZASSERT_EXIT(ptr);

    if (ptr->maxPlayFrames() == 0) {
        //run.
        QMessageBox::information(nullptr, "Zeno", tr("Run the graph before recording"), QMessageBox::Ok);
        return;
    }

    int frameLeft = ptr->beginFrameNumber;
    int frameRight = ptr->endFrameNumber;

    ZRecordVideoDlg dlg(frameLeft, frameRight, this);
    if (QDialog::Accepted == dlg.exec())
    {
        int frameStart = 0, frameEnd = 0, fps = 0, bitrate = 0, width = 0, height = 0;
        QString presets, path, filename;
        dlg.getInfo(frameStart, frameEnd, fps, bitrate, presets, width, height, path, filename);
        //validation.

        VideoRecInfo recInfo;
        recInfo.record_path = path;
        recInfo.frameRange = { frameStart, frameEnd };
        recInfo.res = { (float)width, (float)height };
        recInfo.bitrate = bitrate;
        recInfo.fps = fps;
        recInfo.videoname = filename;

        Zenovis::GetInstance().startPlay(true);

        RecordVideoMgr recordMgr(m_view, recInfo, nullptr);
        ZRecordProgressDlg dlgProc(recInfo);
        connect(&recordMgr, SIGNAL(frameFinished(int)), &dlgProc, SLOT(onFrameFinished(int)));
        connect(&recordMgr, SIGNAL(recordFinished()), &dlgProc, SLOT(onRecordFinished()));
        connect(&recordMgr, SIGNAL(recordFailed(QString)), &dlgProc, SLOT(onRecordFailed(QString)));
        if (QDialog::Accepted == dlgProc.exec())
        {
        }
        else
        {
            recordMgr.cancelRecord();
        }
    }
}

void DisplayWidget::onKill()
{
    killProgram();
}
