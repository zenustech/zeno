#include <zenovis/RenderEngine.h>
#include "viewportwidget.h"
#include "zenovis.h"
#include "camerakeyframe.h"
#include "timeline/ztimeline.h"
#include <zenomodel/include/graphsmanagment.h>
#include "launch/corelaunch.h"
#include "zenomainwindow.h"
#include "dialog/zrecorddlg.h"
#include "dialog/zrecprogressdlg.h"
#include <zeno/utils/log.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/funcs/ObjectGeometryInfo.h>
#include <util/log.h>
#include <zenoui/style/zenostyle.h>
//#include <zeno/utils/zeno_p.h>
// #include <nodesys/nodesmgr.h>
#include <zenomodel/include/nodesmgr.h>
#include <cmath>
#include <algorithm>
#include <optional>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zenomodel/include/uihelper.h>
#include "recordvideomgr.h"
#define CMP(x, y) \
	(fabsf(x - y) <= FLT_EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

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

static std::optional<float> ray_box_intersect(
    zeno::vec3f const &bmin,
    zeno::vec3f const &bmax,
    zeno::vec3f const &ray_pos,
    zeno::vec3f const &ray_dir
) {
    //objectGetBoundingBox(IObject *ptr, vec3f &bmin, vec3f &bmax);

    auto &min = bmin;
    auto &max = bmax;
    auto &p = ray_pos;
    auto &d = ray_dir;
    //auto &t = t;

    float t1 = (min[0] - p[0]) / (CMP(d[0], 0.0f) ? 0.00001f : d[0]);
    float t2 = (max[0] - p[0]) / (CMP(d[0], 0.0f) ? 0.00001f : d[0]);
    float t3 = (min[1] - p[1]) / (CMP(d[1], 0.0f) ? 0.00001f : d[1]);
    float t4 = (max[1] - p[1]) / (CMP(d[1], 0.0f) ? 0.00001f : d[1]);
    float t5 = (min[2] - p[2]) / (CMP(d[2], 0.0f) ? 0.00001f : d[2]);
    float t6 = (max[2] - p[2]) / (CMP(d[2], 0.0f) ? 0.00001f : d[2]);

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    // if tmax < 0, ray is intersecting AABB
    // but entire AABB is behing it's origin
    if (tmax < 0) {
        return std::nullopt;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax) {
        return std::nullopt;
    }

    float t_result = tmin;

    // If tmin is < 0, tmax is closer
    if (tmin < 0.0f) {
        t_result = tmax;
    }
    //zeno::vec3f  final_t = p + d * t_result;
    return t_result;
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

void CameraControl::setAperture(float aperture){
    m_aperture = aperture;
}
void CameraControl::setDisPlane(float disPlane){
    m_focalPlaneDistance = disPlane;
}

void CameraControl::fakeMousePressEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::MiddleButton) {
        m_lastPos = event->pos();
    }
    else if (event->buttons() & Qt::LeftButton) {
        m_boundRectStartPos = event->pos();
        // check if clicked a selected object
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        auto front = scene->camera->m_lodfront;
        auto dir = screenToWorldRay(event->x() / res().x(), event->y() / res().y());
        if (!scene->selected.empty() && transformer->isTransformMode() && transformer->clickedAnyHandler(realPos(), dir, front)) {
            transformer->startTransform();
        }
    }
}

void CameraControl::lookTo(int dir) {
    if (dir < 0 || dir > 5) return;
    auto x_axis = QVector3D(1, 0, 0);
    auto y_axis = QVector3D(0, 1, 0);
    auto z_axis = QVector3D(0, 0, 1);
    switch (dir) {
    case 0:
        // front view
        m_theta = 0.f; m_phi = 0.f;
        Zenovis::GetInstance().updateCameraFront(m_center + z_axis * m_radius, -z_axis, y_axis);
        break;
    case 1:
        // right view
        m_theta = 0.0f; m_phi = - glm::pi<float>() / 2.f;
        Zenovis::GetInstance().updateCameraFront(m_center + x_axis * m_radius, -x_axis, y_axis);
        break;
    case 2:
        // top view
        m_theta = - glm::pi<float>() / 2; m_phi = 0.f;
        Zenovis::GetInstance().updateCameraFront(m_center + y_axis * m_radius, -z_axis, y_axis);
        break;
    case 3:
        // back view
        m_theta = 0.f; m_phi = glm::pi<float>();
        Zenovis::GetInstance().updateCameraFront(m_center - z_axis * m_radius, z_axis, y_axis);
        break;
    case 4:
        // left view
        m_theta = 0.f; m_phi = glm::pi<float>() / 2.f;
        Zenovis::GetInstance().updateCameraFront(m_center - x_axis * m_radius, x_axis, y_axis);
        break;
    case 5:
        // bottom view
        m_theta = glm::pi<float>() / 2; m_phi = 0.f;
        Zenovis::GetInstance().updateCameraFront(m_center - y_axis * m_radius, y_axis, z_axis);
        break;
    default:
        break;
    }
    m_ortho_mode = true;
    updatePerspective();
    m_ortho_mode = false;
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::clearTransformer() {
    transformer->clear();
}

void CameraControl::changeTransformOperation(int mode) {
    switch (mode) {
    case 0:
        transformer->toTranslate();
        break;
    case 1:
        transformer->toRotate();
        break;
    case 2:
        transformer->toScale();
        break;
    default:
        break;
    }
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformCoordSys() {
    transformer->changeCoordSys();
    zenoApp->getMainWindow()->updateViewport();
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
            auto dir = screenToWorldRay(event->pos().x() / res().x(), event->pos().y() / res().y());
            auto camera_pos = realPos();
            auto x = event->x() * 1.0f;
            auto y = event->y() * 1.0f;
            x = (2 * x / res().x()) - 1;
            y = 1 - (2 * y / res().y());
            auto mouse_pos = glm::vec2(x, y);
            auto vp = scene->camera->m_proj * scene->camera->m_view;
            transformer->transform(camera_pos, mouse_pos, dir, scene->camera->m_lodfront, vp);
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

    if(zenoApp->getMainWindow()->lightPanel != nullptr){
        zenoApp->getMainWindow()->lightPanel->camApertureEdit->setText(QString::number(m_aperture));
        zenoApp->getMainWindow()->lightPanel->camDisPlaneEdit->setText(QString::number(m_focalPlaneDistance));
    }
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
            bool moved = false;
            if (m_boundRectStartPos != event->pos()) {
                // create/modify transform primitive node
                moved = true;
            }
            transformer->endTransform(moved);
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
                    zeno::vec3f ro(cam_pos[0], cam_pos[1], cam_pos[2]);
                    zeno::vec3f rd(rdir[0], rdir[1], rdir[2]);
                    zeno::vec3f bmin,bmax;
                    if (zeno::objectGetBoundingBox(ptr, bmin, bmax) ){
                        if (auto ret = ray_box_intersect(bmin, bmax, ro, rd)) {
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

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        mainWin->onPrimitiveSelected(scene->selected);
    }
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

    // https://blog.csdn.net/zhujiangm/article/details/90760744
    // https://blog.csdn.net/jays_/article/details/83783871
    setFocusPolicy(Qt::ClickFocus);

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

void ViewportWidget::cameraLookTo(int dir) {
     m_camera->lookTo(dir);
}


void ViewportWidget::clearTransformer() {
    m_camera->clearTransformer();
}

void ViewportWidget::changeTransformOperation(int mode) {
    m_camera->changeTransformOperation(mode);
}

void ViewportWidget::changeTransformCoordSys() {
    m_camera->changeTransformCoordSys();
}

void ViewportWidget::updateCameraProp(float aperture, float disPlane) {
    m_camera->setAperture(aperture);
    m_camera->setDisPlane(disPlane);
    updatePerspective();
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
    // viewport interaction need to set mouse tracking true
    // but it will lead to a light panel edit bug
    // m_view->setMouseTracking(true);
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

void DisplayWidget::resetTimeline(TIMELINE_INFO info)
{
    BlockSignalScope scope(m_timeline);
    m_timeline->setAlways(info.bAlways);
    m_timeline->initFromTo(info.beginFrame, info.endFrame);
    m_timeline->setSliderValue(info.currFrame);
}

ViewportWidget* DisplayWidget::getViewportWidget()
{
    return m_view;
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

    m_view->clearTransformer();
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

void ViewportWidget::keyPressEvent(QKeyEvent* event) {
    _base::keyPressEvent(event);
    //qInfo() << event->key();
    if (event->key() == Qt::Key_T)
        this->changeTransformOperation(0);
    if (event->key() == Qt::Key_R)
        this->changeTransformOperation(1);
    if (event->key() == Qt::Key_E)
        this->changeTransformOperation(2);
    if (event->key() == Qt::Key_M)
        this->changeTransformCoordSys();

    if (event->key() == Qt::Key_1)
        this->cameraLookTo(0);
    if (event->key() == Qt::Key_3)
        this->cameraLookTo(1);
    if (event->key() == Qt::Key_7)
        this->cameraLookTo(2);

    bool ctrl_pressed = event->modifiers() & Qt::ControlModifier;
    if (ctrl_pressed && event->key() == Qt::Key_1)
        this->cameraLookTo(3);
    if (ctrl_pressed && event->key() == Qt::Key_3)
        this->cameraLookTo(4);
    if (ctrl_pressed && event->key() == Qt::Key_7)
        this->cameraLookTo(5);
}

void ViewportWidget::keyReleaseEvent(QKeyEvent* event) {
    _base::keyReleaseEvent(event);
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
