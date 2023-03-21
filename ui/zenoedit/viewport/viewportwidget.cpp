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
#include <zenovis/DrawOptions.h>
#include <zeno/funcs/ObjectGeometryInfo.h>
#include <util/log.h>
#include <zenoui/style/zenostyle.h>
#include <nodesview/zenographseditor.h>
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
#include <zeno/types/UserData.h>


#define ENABLE_RECORD_PROGRESS_DIG


using std::string;
using std::unordered_set;
using std::unordered_map;
CameraControl::CameraControl(ViewportWidget* parent)
    : QObject(parent)
    , m_mmb_pressed(false)
    , m_theta(0.)
    , m_phi(0.)
    , m_ortho_mode(false)
    , m_fov(45.)
    , m_radius(5.0)
    , m_res(1, 1)
    , m_aperture(0.1f)
    , m_focalPlaneDistance(2.0f)
{
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

ViewportWidget* CameraControl::getViewport() const
{
    return qobject_cast<ViewportWidget*>(parent());
}

void CameraControl::fakeMousePressEvent(QMouseEvent* event)
{
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    Zenovis* pZenovis = pViewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);
    auto scene = pZenovis->getSession()->get_scene();
    if (scene->camera->m_need_sync) {
        m_center = {
            scene->camera->m_zxx_in.cx,
            scene->camera->m_zxx_in.cy,
            scene->camera->m_zxx_in.cz
        };
        m_theta = scene->camera->m_zxx_in.theta;
        m_phi = scene->camera->m_zxx_in.phi;
        m_radius = scene->camera->m_zxx_in.radius;
        m_fov = scene->camera->m_fov;
        m_aperture = scene->camera->m_aperture;
        m_focalPlaneDistance = scene->camera->focalPlaneDistance;
        scene->camera->m_need_sync = false;
    }
    if (event->buttons() & Qt::MiddleButton) {
        m_lastPos = event->pos();
    }
    else if (event->buttons() & Qt::LeftButton) {
        m_boundRectStartPos = event->pos();
        // check if clicked a selected object
        auto front = scene->camera->m_lodfront;
        auto dir = screenToWorldRay(event->x() / res().x(),
                                    event->y() / res().y());

        auto transformer = pViewport->fakeTransformer();
        ZASSERT_EXIT(transformer);

        if (!scene->selected.empty() &&
            transformer->isTransformMode() &&
            transformer->clickedAnyHandler(realPos(), dir, front)) {
            transformer->startTransform();
        }
    }
}

void CameraControl::lookTo(int dir) {
    if (dir < 0 || dir > 6) return;
    auto x_axis = QVector3D(1, 0, 0);
    auto y_axis = QVector3D(0, 1, 0);
    auto z_axis = QVector3D(0, 0, 1);

    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    Zenovis *pZenovis = pViewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);

    switch (dir) {
    case 0:
        // front view
        m_theta = 0.f; m_phi = 0.f;
        pZenovis->updateCameraFront(m_center + z_axis * m_radius, -z_axis, y_axis);
        break;
    case 1:
        // right view
        m_theta = 0.0f; m_phi = - glm::pi<float>() / 2.f;
        pZenovis->updateCameraFront(m_center + x_axis * m_radius, -x_axis, y_axis);
        break;
    case 2:
        // top view
        m_theta = - glm::pi<float>() / 2; m_phi = 0.f;
        pZenovis->updateCameraFront(m_center + y_axis * m_radius, -z_axis, y_axis);
        break;
    case 3:
        // back view
        m_theta = 0.f; m_phi = glm::pi<float>();
        pZenovis->updateCameraFront(m_center - z_axis * m_radius, z_axis, y_axis);
        break;
    case 4:
        // left view
        m_theta = 0.f; m_phi = glm::pi<float>() / 2.f;
        pZenovis->updateCameraFront(m_center - x_axis * m_radius, x_axis, y_axis);
        break;
    case 5:
        // bottom view
        m_theta = glm::pi<float>() / 2; m_phi = 0.f;
        pZenovis->updateCameraFront(m_center - y_axis * m_radius, y_axis, z_axis);
        break;
    case 6:
        // back to origin
        m_center = {0, 0, 0};
        m_radius = 5.f;
        m_theta = 0.f; m_phi = 0.f;
        pZenovis->updateCameraFront(m_center, -z_axis, y_axis);
    default:
        break;
    }
    m_ortho_mode = true;
    updatePerspective();
    m_ortho_mode = false;
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::clearTransformer() {
    auto transformer = getViewport()->fakeTransformer();
    ZASSERT_EXIT(transformer);
    transformer->clear();
}

void CameraControl::changeTransformOperation(const QString& node)
{
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    auto transformer = pViewport->fakeTransformer();
    ZASSERT_EXIT(transformer);

    auto opt = transformer->getTransOpt();
    transformer->clear();

    Zenovis* pZenovis = pViewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);

    auto scene = pZenovis->getSession()->get_scene();
    for (auto const &[key, _] : scene->objectsMan->pairs()) {
        if (key.find(node.toStdString()) != std::string::npos) {
            scene->selected.insert(key);
            transformer->addObject(key);
        }
    }
    transformer->setTransOpt(opt);
    transformer->changeTransOpt();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformOperation(int mode)
{
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    auto transformer = pViewport->fakeTransformer();
    ZASSERT_EXIT(transformer);

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

void CameraControl::changeTransformCoordSys()
{
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    auto transformer = pViewport->fakeTransformer();
    ZASSERT_EXIT(transformer);

    transformer->changeCoordSys();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::resizeTransformHandler(int dir)
{
    ViewportWidget *pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    auto transformer = pViewport->fakeTransformer();
    ZASSERT_EXIT(transformer);

    transformer->resizeHandler(dir);
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::fakeMouseMoveEvent(QMouseEvent* event)
{
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    Zenovis* pZenovis = pViewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);

    auto session = pZenovis->getSession();
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
        auto transformer = pViewport->fakeTransformer();
        ZASSERT_EXIT(transformer);
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
    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    Zenovis* pZenovis = pViewport->getZenoVis();
    ZASSERT_EXIT(pZenovis);
    pZenovis->updatePerspective(m_res, PerspectiveInfo(cx, cy, cz, m_theta, m_phi, m_radius, m_fov, m_ortho_mode,
                                                      m_aperture, m_focalPlaneDistance));
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

void CameraControl::fakeMouseDoubleClickEvent(QMouseEvent* event) {
    auto pos = event->pos();

    ViewportWidget* pViewport = getViewport();
    ZASSERT_EXIT(pViewport);
    auto picker = pViewport->picker();
    ZASSERT_EXIT(picker);

    auto picked_prim = picker->just_pick_prim(pos.x(), pos.y());
    if (!picked_prim.empty()) {
        auto obj_node_location = zeno::NodeSyncMgr::GetInstance().searchNodeOfPrim(picked_prim);
        auto subgraph_name = obj_node_location->subgraph.data(ROLE_OBJNAME).toString();
        auto obj_node_name = obj_node_location->node.data(ROLE_OBJID).toString();
        ZenoMainWindow* pWin = zenoApp->getMainWindow();
        if (pWin) {
            ZenoGraphsEditor* pEditor = pWin->getAnyEditor();
            if (pEditor)
                pEditor->activateTab(subgraph_name, "", obj_node_name);
        }
    }
}
//void CameraControl::fakeMouseDoubleClickEvent(QMouseEvent* event) {
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
        ViewportWidget* pViewport = getViewport();
        ZASSERT_EXIT(pViewport);
        Zenovis *pZenovis = pViewport->getZenoVis();
        ZASSERT_EXIT(pZenovis);

        auto scene = pZenovis->getSession()->get_scene();
        auto picker = pViewport->picker();
        ZASSERT_EXIT(picker);
        auto transformer = pViewport->fakeTransformer();
        ZASSERT_EXIT(transformer);

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

            auto onPrimSelected = [pZenovis]() {
                auto scene = pZenovis->getSession()->get_scene();
                ZenoMainWindow* mainWin = zenoApp->getMainWindow();
                mainWin->onPrimitiveSelected(scene->selected);
            };

            QPoint releasePos = event->pos();
            if (m_boundRectStartPos == releasePos) {
                if (picker->is_draw_mode()) {
                    // zeno::log_info("res_w: {}, res_h: {}", res()[0], res()[1]);
                    picker->pick_depth(releasePos.x(), releasePos.y());
                }
                else {
                    picker->pick(releasePos.x(), releasePos.y());
                    picker->sync_to_scene();
                    if (scene->select_mode == zenovis::PICK_OBJECT)
                        onPrimSelected();
                    transformer->clear();
                    transformer->addObject(picker->get_picked_prims());
                }
            } else {
                int x0 = m_boundRectStartPos.x();
                int y0 = m_boundRectStartPos.y();
                int x1 = releasePos.x();
                int y1 = releasePos.y();
                picker->pick(x0, y0, x1, y1);
                picker->sync_to_scene();
                if (scene->select_mode == zenovis::PICK_OBJECT)
                    onPrimSelected();
                transformer->clear();
                transformer->addObject(picker->get_picked_prims());
            }
    }
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
    : QGLWidget(parent)
    , m_camera(nullptr)
    , updateLightOnce(true)
    , m_pauseRenderDally(new QTimer)
    , m_wheelEventDally(new QTimer)
    , simpleRenderChecked(false)
    , m_bMovingCamera(false)
    , m_zenovis(nullptr)
{
    m_zenovis = new Zenovis(this);
    m_picker = std::make_shared<zeno::Picker>(this);
    m_fakeTrans = std::make_shared<zeno::FakeTransformer>(this);

    QGLFormat fmt;
    int nsamples = 16;  // TODO: adjust in a zhouhang-panel
    fmt.setSamples(nsamples);
    fmt.setVersion(3, 2);
    fmt.setProfile(QGLFormat::CoreProfile);
    setFormat(fmt);

    // https://blog.csdn.net/zhujiangm/article/details/90760744
    // https://blog.csdn.net/jays_/article/details/83783871
    setFocusPolicy(Qt::ClickFocus);

    m_camera = new CameraControl(this);
    m_zenovis->m_camera_control = m_camera;

    connect(m_zenovis, &Zenovis::objectsUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin)
            emit mainWin->visObjectsUpdated(this, frameid);
    });

    connect(m_zenovis, &Zenovis::frameUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin)
            mainWin->visFrameUpdated(frameid);
    });

    connect(m_pauseRenderDally, &QTimer::timeout, [&](){
        auto scene = m_zenovis->getSession()->get_scene();
        scene->drawOptions->simpleRender = false;
        scene->drawOptions->needRefresh = true;
        m_pauseRenderDally->stop();
        //std::cout << "SR: SimpleRender false, Active " << m_pauseRenderDally->isActive() << "\n";
    });

    connect(m_wheelEventDally, &QTimer::timeout, [&](){
        m_wheelEventDally->stop();
        m_bMovingCamera = false;
    });
}

void ViewportWidget::setSimpleRenderOption() {
    if(simpleRenderChecked)
        return;

    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
    m_pauseRenderDally->stop();
    m_pauseRenderDally->start(4000);
}

ViewportWidget::~ViewportWidget()
{
    testCleanUp();
    delete m_pauseRenderDally;
    delete m_wheelEventDally;
}

void ViewportWidget::testCleanUp()
{
    delete m_camera;
    delete m_zenovis;
    m_camera = nullptr;
    m_zenovis = nullptr;
    m_picker.reset();
    m_fakeTrans.reset();
}

namespace {
struct OpenGLProcAddressHelper {
    inline static QGLContext *ctx;

    static void *getProcAddress(const char *name) {
        return (void *)ctx->getProcAddress(name);
    }
};
}

void ViewportWidget::initializeGL()
{
    OpenGLProcAddressHelper::ctx = context();
    ZASSERT_EXIT(m_zenovis);
    m_zenovis->loadGLAPI((void *)OpenGLProcAddressHelper::getProcAddress);
    m_zenovis->initializeGL();
    ZASSERT_EXIT(m_picker);
    m_picker->initialize();
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

Zenovis* ViewportWidget::getZenoVis() const
{
    return m_zenovis;
}

std::shared_ptr<zeno::Picker> ViewportWidget::picker() const
{
    return m_picker;
}

std::shared_ptr<zeno::FakeTransformer> ViewportWidget::fakeTransformer() const
{
    return m_fakeTrans;
}

zenovis::Session* ViewportWidget::getSession() const
{
    return m_zenovis->getSession();
}

bool ViewportWidget::isPlaying() const
{
    return m_zenovis->isPlaying();
}

void ViewportWidget::startPlay(bool bPlaying)
{
    m_zenovis->startPlay(bPlaying);
}

int ViewportWidget::getCurrentFrameId()
{
    return m_zenovis->getCurrentFrameId();
}

int ViewportWidget::setCurrentFrameId(int frameid)
{
    return m_zenovis->setCurrentFrameId(frameid);
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
    m_zenovis->paintGL();
    if(updateLightOnce){
        auto scene = m_zenovis->getSession()->get_scene();

        if(scene->objectsMan->lightObjects.size() > 0){
            zenoApp->getMainWindow()->updateLightList();
            updateLightOnce = false;
        }
    }
}

void ViewportWidget::mousePressEvent(QMouseEvent* event)
{
    if(event->button() == Qt::MidButton){
        m_bMovingCamera = true;
        setSimpleRenderOption();
    }
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* event)
{
    if(event->button() == Qt::MidButton){
        m_bMovingCamera = true;
    }
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ViewportWidget::wheelEvent(QWheelEvent* event)
{
    m_bMovingCamera = true;
    m_wheelEventDally->start(100);
    setSimpleRenderOption();

    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}

void ViewportWidget::mouseReleaseEvent(QMouseEvent *event) {
    if(event->button() == Qt::MidButton){
        m_bMovingCamera = false;
    }
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event); 
    update();
}

void ViewportWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseDoubleClickEvent(event);
    update();
}
//void ViewportWidget::mouseDoubleClickEvent(QMouseEvent* event) {
void ViewportWidget::cameraLookTo(int dir) {
     m_camera->lookTo(dir);
}


void ViewportWidget::clearTransformer() {
    m_camera->clearTransformer();
}

void ViewportWidget::changeTransformOperation(const QString& node) {
    m_camera->changeTransformOperation(node);
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

DisplayWidget::DisplayWidget(QWidget* parent)
    : QWidget(parent)
    , m_view(nullptr)
    , m_pTimer(nullptr)
    , m_bRecordRun(false)
{
    QVBoxLayout* pLayout = new QVBoxLayout;
    pLayout->setContentsMargins(0, 0, 0, 0);
    pLayout->setSpacing(0);

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    initRecordMgr();

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

    setLayout(pLayout);

    m_camera_keyframe = new CameraKeyframeWidget;
    Zenovis* pZenovis = m_view->getZenoVis();
    if (pZenovis)
    {
        pZenovis->m_camera_keyframe = m_camera_keyframe;
    }
    //connect(m_view, SIGNAL(sig_Draw()), this, SLOT(onRun()));

	m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

DisplayWidget::~DisplayWidget()
{

}

void DisplayWidget::initRecordMgr()
{
    m_recordMgr.setParent(this);

    connect(&m_recordMgr, &RecordVideoMgr::frameFinished, this, [=](int frameid) {
        zeno::log_info("frame {} has been recorded", frameid);
    });

#ifndef ENABLE_RECORD_PROGRESS_DIG
    connect(&m_recordMgr, &RecordVideoMgr::recordFinished, this, [=](QString recPath) {
        VideoRecInfo _recInfo;
        _recInfo.record_path = recPath;
        ZRecordProgressDlg dlgProc(_recInfo);
        dlgProc.onRecordFinished();
        dlgProc.exec();
    });
#endif
}

void DisplayWidget::testCleanUp()
{
    m_view->testCleanUp();
}

void DisplayWidget::init()
{
    //m_camera->installEventFilter(this);
}

ViewportWidget* DisplayWidget::getViewportWidget()
{
    return m_view;
}

QSize DisplayWidget::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(QSize(12, 400));
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
    m_view->startPlay(bChecked);
}

void DisplayWidget::updateFrame(const QString &action) // cihou optix
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    if (mainWin && mainWin->inDlgEventLoop())
        return;

    if (action == "newFrame") {
        m_pTimer->stop();
        //zeno::log_warn("stop");
        return;
    } else if (action == "finishFrame") {
        bool bPlaying = m_view->isPlaying();
        if (isOptxRendering() || bPlaying) {
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

void DisplayWidget::onCommandDispatched(int actionType, bool bChecked)
{
    if (actionType == ZenoMainWindow::ACTION_SMOOTH_SHADING)
    {
        m_view->getSession()->set_smooth_shading(bChecked);
        updateFrame("");
    }
    else if (actionType == ZenoMainWindow::ACTION_NORMAL_CHECK)
    {
        m_view->getSession()->set_normal_check(bChecked);
        updateFrame("");
    }
    else if (actionType == ZenoMainWindow::ACTION_WIRE_FRAME)
    {
        m_view->getSession()->set_render_wireframe(bChecked);
        updateFrame("");
    }
    else if (actionType == ZenoMainWindow::ACTION_SHOW_GRID)
    {
        m_view->getSession()->set_show_grid(bChecked);
        //todo: need a notify mechanism from zenovis/session.
        updateFrame("");
    }
    else if (actionType == ZenoMainWindow::ACTION_BACKGROUND_COLOR)
    {
        auto [r, g, b] = m_view->getSession()->get_background_color();
        auto c = QColor::fromRgbF(r, g, b);
        c = QColorDialog::getColor(c);
        if (c.isValid()) {
            m_view->getSession()->set_background_color(c.redF(), c.greenF(), c.blueF());
            updateFrame("");
        }
    }
    else if (actionType == ZenoMainWindow::ACTION_SOLID)
    {
        const char* e = "bate";
        m_view->getSession()->set_render_engine(e);
        updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_SHADING)
    {
        const char* e = "zhxx";
        m_view->getSession()->set_render_engine(e);
        //m_view->getSession()->set_enable_gi(false);
        updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_OPTIX)
    {
        const char* e = "optx";
        m_view->getSession()->set_render_engine(e);
        updateFrame(QString::fromUtf8(e));
    }
    else if (actionType == ZenoMainWindow::ACTION_NODE_CAMERA)
    {
        int frameid = m_view->getSession()->get_curr_frameid();
        auto *scene = m_view->getSession()->get_scene();
        for (auto const& [key, ptr] : scene->objectsMan->pairs()) {
            if (key.find("MakeCamera") != std::string::npos && key.find(zeno::format(":{}:", frameid)) != std::string::npos) {
                auto cam = dynamic_cast<zeno::CameraObject*>(ptr)->get();
                scene->camera->setCamera(cam);
                updateFrame();
            }
        }
    } 
    else if (actionType == ZenoMainWindow::ACTION_RECORD_VIDEO) 
    {
        onRecord();
    }
    else if (actionType == ZenoMainWindow::ACTION_SCREEN_SHOOT)
    {
        onScreenShoot();
    }
    else if (actionType == ZenoMainWindow::ACTION_BLACK_WHITE 
        || actionType == ZenoMainWindow::ACTION_GREEK
        || actionType == ZenoMainWindow::ACTION_DAY_LIGHT 
        || actionType == ZenoMainWindow::ACTION_DEFAULT 
        || actionType == ZenoMainWindow::ACTION_FOOTBALL_FIELD 
        || actionType == ZenoMainWindow::ACTION_FOREST 
        || actionType == ZenoMainWindow::ACTION_LAKE 
        || actionType == ZenoMainWindow::ACTION_SEA) 
    {
        //todo: no implementation from master.
    }
    //record: todo: wait for merge from branch master/tmp-recordvideo.
}

void DisplayWidget::onFinished()
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    int frameid_ui = timeline->value();
    if (frameid_ui != m_view->getCurrentFrameId())
    {
        m_view->setCurrentFrameId(frameid_ui);
        updateFrame();
        onPlayClicked(false);
        BlockSignalScope scope(timeline);
        timeline->setPlayButtonChecked(false);
    }
}

bool DisplayWidget::isOptxRendering() const
{
    auto sess = m_view->getSession();
    if (!sess)
        return false;
    auto scene = sess->get_scene();
    if (!scene)
        return false;

    return (scene->renderMan && scene->renderMan->getDefaultEngineName() == "optx");
}

void DisplayWidget::onSliderValueChanged(int frame)
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    mainWin->clearErrorMark();

    ZTimeline* timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    if (mainWin->isAlways())
    {
        auto pGraphsMgr = zenoApp->graphsManagment();
        IGraphsModel* pModel = pGraphsMgr->currentModel();
        if (!pModel)
            return;
        launchProgram(pModel, frame, frame);
    }
    else
    {
        m_view->setCurrentFrameId(frame);
        updateFrame();
        onPlayClicked(false);
        BlockSignalScope scope(timeline);
        timeline->setPlayButtonChecked(false);
    }
}

void DisplayWidget::beforeRun()
{
    m_view->clearTransformer();
    m_view->getSession()->get_scene()->selected.clear();
}

void DisplayWidget::afterRun()
{
    m_view->updateLightOnce = true;
    auto scene = m_view->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::onRun(int frameStart, int frameEnd)
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);

    auto pGraphsMgr = zenoApp->graphsManagment();
    ZASSERT_EXIT(pGraphsMgr);

    IGraphsModel* pModel = pGraphsMgr->currentModel();
    ZASSERT_EXIT(pModel);

    mainWin->clearErrorMark();

    m_view->clearTransformer();
    m_view->getSession()->get_scene()->selected.clear();

    launchProgram(pModel, frameStart, frameEnd);

    m_view->updateLightOnce = true;
    auto scene = m_view->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::onRun()
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    mainWin->clearErrorMark();

    m_view->clearTransformer();
    m_view->getSession()->get_scene()->selected.clear();

    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);
    QPair<int, int> fromTo = timeline->fromTo();
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
    auto scene = m_view->getSession()->get_scene();
    scene->objectsMan->lightObjects.clear();
}

void DisplayWidget::runAndRecord(const VideoRecInfo& recInfo)
{
    //reset the record info first.
    m_bRecordRun = true;
    m_recordMgr.setRecordInfo(recInfo);

    m_view->startPlay(true);

    //and then play.
    onPlayClicked(true);

    //run first.
    onRun();

    if (recInfo.exitWhenRecordFinish)
    {
        connect(&m_recordMgr, &RecordVideoMgr::recordFinished, this, [=]() {
            zenoApp->quit();
        });
    }
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
    if (event->key() == Qt::Key_0)
        this->cameraLookTo(6);

    bool ctrl_pressed = event->modifiers() & Qt::ControlModifier;
    if (ctrl_pressed && event->key() == Qt::Key_1)
        this->cameraLookTo(3);
    if (ctrl_pressed && event->key() == Qt::Key_3)
        this->cameraLookTo(4);
    if (ctrl_pressed && event->key() == Qt::Key_7)
        this->cameraLookTo(5);

    if (event->key() == Qt::Key_Backspace)
        m_camera->resizeTransformHandler(0);
    if (event->key() == Qt::Key_Plus)
        m_camera->resizeTransformHandler(1);
    if (event->key() == Qt::Key_Minus)
        m_camera->resizeTransformHandler(2);
}

void ViewportWidget::keyReleaseEvent(QKeyEvent* event) {
    _base::keyReleaseEvent(event);
}

void DisplayWidget::onScreenShoot()
{
    QString path = QFileDialog::getSaveFileName(
        nullptr, tr("Path to Save"), "",
        tr("PNG images(*.png);;JPEG images(*.jpg);;BMP images(*.bmp);;EXR images(*.exr);;HDR images(*.hdr);;"));
    QString ext = QFileInfo(path).suffix();
    if (ext.isEmpty()) {
        //qt bug: won't fill extension automatically.
        ext = "png";
        path.append(".png");
    }
    if (!path.isEmpty())
    {
        ZASSERT_EXIT(m_view);
        m_view->getSession()->do_screenshot(path.toStdString(), ext.toStdString());
    }
}

void DisplayWidget::onRecord()
{
    auto& pGlobalComm = zeno::getSession().globalComm;
    ZASSERT_EXIT(pGlobalComm);

    int frameLeft = 0, frameRight = 0;
    if (pGlobalComm->maxPlayFrames() > 0)
    {
        frameLeft = pGlobalComm->beginFrameNumber;
        frameRight = pGlobalComm->endFrameNumber;
    }
    else
    {
        frameLeft = 0;
        frameRight = 0;
    }

    ZRecordVideoDlg dlg(frameLeft, frameRight, this);
    if (QDialog::Accepted == dlg.exec())
    {
        VideoRecInfo recInfo;
        dlg.getInfo(
                recInfo.frameRange.first,
                recInfo.frameRange.second,
                recInfo.fps,
                recInfo.bitrate,
                recInfo.res[0],
                recInfo.res[1],
                recInfo.record_path,
                recInfo.videoname,
                recInfo.numOptix,
                recInfo.numMSAA,
                recInfo.bRecordAfterRun,
                recInfo.bExportVideo
            );
        //validation.

        m_recordMgr.setRecordInfo(recInfo);

        bool bRun = !recInfo.bRecordAfterRun;

        if (!bRun && pGlobalComm->maxPlayFrames() == 0)
        {
            QMessageBox::information(nullptr, "Zeno", tr("Run the graph before recording"), QMessageBox::Ok);
            return;
        }

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);

        int recStartFrame = recInfo.frameRange.first;
        int recEndFrame = recInfo.frameRange.second;

        if (bRun)
        {
            //clear the global Comm first, to avoid play old frames.
            zeno::getSession().globalComm->clearState();

            //expand the timeline if necessary.
            ZTimeline* timeline = mainWin->timeline();
            auto pair = timeline->fromTo();
            if (pair.first > recStartFrame || pair.second < recEndFrame)
            {
                //expand timeline
                timeline->initFromTo(qMin(pair.first, recStartFrame), qMax(recEndFrame, pair.second));
            }

            //reset the current frame on timeline.
            moveToFrame(recStartFrame);

            // and then toggle play.
            mainWin->toggleTimelinePlay(true);

            //and then run.
            onRun(recInfo.frameRange.first, recInfo.frameRange.second);
        }
        else
        {
            // first, set the time frame start end.
            moveToFrame(recStartFrame);
            // and then play.
            mainWin->toggleTimelinePlay(true);

#ifdef ENABLE_RECORD_PROGRESS_DIG
            ZRecordProgressDlg dlgProc(recInfo);
            connect(&m_recordMgr, SIGNAL(frameFinished(int)), &dlgProc, SLOT(onFrameFinished(int)));
            connect(&m_recordMgr, SIGNAL(recordFinished(QString)), &dlgProc, SLOT(onRecordFinished(QString)));
            connect(&m_recordMgr, SIGNAL(recordFailed(QString)), &dlgProc, SLOT(onRecordFailed(QString)));
            connect(&dlgProc, SIGNAL(cancelTriggered()), &m_recordMgr, SLOT(cancelRecord()));
            connect(&dlgProc, &ZRecordProgressDlg::pauseTriggered, this, [=]() { mainWin->toggleTimelinePlay(false); });
            connect(&dlgProc, &ZRecordProgressDlg::continueTriggered, this, [=]() { mainWin->toggleTimelinePlay(true); });

            if (QDialog::Accepted == dlgProc.exec())
            {
            } else
            {
                m_recordMgr.cancelRecord();
            }
#endif
        }
    }
}

void DisplayWidget::moveToFrame(int frame)
{
    ZenoMainWindow *mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    ZTimeline *timeline = mainWin->timeline();
    ZASSERT_EXIT(timeline);

    m_view->setCurrentFrameId(frame);
    updateFrame();
    onPlayClicked(false);
    {
        BlockSignalScope scope(timeline);
        timeline->setPlayButtonChecked(false);
        timeline->setSliderValue(frame);
    }
}

void DisplayWidget::onKill()
{
    killProgram();
}

void DisplayWidget::onNodeSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select) {
    // tmp code for Primitive Filter Node interaction
    if (nodes.size() > 1) return;
    auto node_id = nodes[0].data(ROLE_OBJNAME).toString();
    if (node_id == "PrimitiveAttrPicker") {
        auto scene = m_view->getSession()->get_scene();
        ZASSERT_EXIT(scene);
        auto picker = m_view->picker();
        ZASSERT_EXIT(picker);
        if (select) {
            // check input nodes
            auto input_nodes = zeno::NodeSyncMgr::GetInstance().getInputNodes(nodes[0], "prim");
            if (input_nodes.size() != 1) return;
            // find prim in object manager
            auto input_node_id = input_nodes[0].get_node_id();
            string prim_name;
            for (const auto &[k, v] : scene->objectsMan->pairsShared()) {
                if (k.find(input_node_id.toStdString()) != string::npos)
                    prim_name = k;
            }
            if (prim_name.empty())
                return;

            zeno::NodeLocation node_location(nodes[0], subgIdx);
            // set callback to picker
            auto callback =
                [node_location, prim_name](unordered_map<string, unordered_set<int>> &picked_elems) -> void {
                std::string picked_elems_str;
                auto &picked_prim_elems = picked_elems[prim_name];
                for (auto elem : picked_prim_elems)
                    picked_elems_str += std::to_string(elem) + ",";
                zeno::NodeSyncMgr::GetInstance().updateNodeParamString(node_location, "selected", picked_elems_str);
            };
            if (picker)
            {
                picker->set_picked_elems_callback(callback);
                // ----- enter node context
                picker->save_context();
            }
            // read selected mode
            auto select_mode_str = zeno::NodeSyncMgr::GetInstance().getInputValString(nodes[0], "mode");
            if (select_mode_str == "triangle") scene->select_mode = zenovis::PICK_MESH;
            else if (select_mode_str == "line") scene->select_mode = zenovis::PICK_LINE;
            else scene->select_mode = zenovis::PICK_VERTEX;
            // read selected elements
            string node_context;
            auto node_selected_str = zeno::NodeSyncMgr::GetInstance().getParamValString(nodes[0], "selected");
            if (!node_selected_str.empty()) {
                auto node_selected_qstr = QString(node_selected_str.c_str());
                auto elements = node_selected_qstr.split(',');
                for (auto& e : elements)
                    if (e.size() > 0) node_context += prim_name + ":" + e.toStdString() + " ";

                if (picker)
                    picker->load_from_str(node_context, scene->select_mode);
            }
            if (picker)
            {
                picker->sync_to_scene();
                picker->focus(prim_name);
            }
        }
        else {
            if (picker)
            {
                picker->load_context();
                picker->sync_to_scene();
                picker->focus("");
                picker->set_picked_elems_callback({});
            }
        }
        zenoApp->getMainWindow()->updateViewport();
    }
    if (node_id == "MakePrimitive") {
        auto picker = m_view->picker();
        ZASSERT_EXIT(picker);
        if (select) {
            picker->switch_draw_mode();
            zeno::NodeLocation node_location(nodes[0], subgIdx);
            auto pick_callback = [nodes, node_location, this](float depth, int x, int y) {
                Zenovis* pZenovis = m_view->getZenoVis();
                ZASSERT_EXIT(pZenovis && pZenovis->getSession());
                auto scene = pZenovis->getSession()->get_scene();
                auto near_ = scene->camera->m_near;
                auto far_ = scene->camera->m_far;
                auto fov = scene->camera->m_fov;
                auto cz = glm::length(scene->camera->m_lodcenter);
                if (depth != 1) {
                    depth = depth * 2 - 1;
                    cz = 2 * near_ * far_ / ((far_ + near_) - depth * (far_ - near_));
                }
                auto w = scene->camera->m_nx;
                auto h = scene->camera->m_ny;
                // zeno::log_info("fov: {}", fov);
                // zeno::log_info("w: {}, h: {}", w, h);
                auto u = (2.0 * x / w) - 1;
                auto v = 1 - (2.0 * y / h);
                // zeno::log_info("u: {}, v: {}", u, v);
                auto cy = v * tan(glm::radians(fov) / 2) * cz;
                auto cx = u * tan(glm::radians(fov) / 2) * w / h * cz;
                // zeno::log_info("cx: {}, cy: {}, cz: {}", cx, cy, -cz);
                glm::vec4 cc = {cx, cy, -cz, 1};
                auto wc = glm::inverse(scene->camera->m_view) * cc;
                wc /= wc.w;
                // zeno::log_info("wx: {}, wy: {}, wz: {}", word_coord.x, word_coord.y, word_coord.z);
                auto points = zeno::NodeSyncMgr::GetInstance().getInputValString(nodes[0], "points");
                zeno::log_info("fetch {}", points.c_str());
                points += std::to_string(wc.x) + " " + std::to_string(wc.y) + " " + std::to_string(wc.z) + " ";
                zeno::NodeSyncMgr::GetInstance().updateNodeInputString(node_location, "points", points);
            };
            picker->set_picked_depth_callback(pick_callback);
        }
        else {
            picker->switch_draw_mode();
        }
    }
}
