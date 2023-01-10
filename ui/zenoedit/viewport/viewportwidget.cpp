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
        auto dir = screenToWorldRay(event->x() / res().x(),
                                    event->y() / res().y());
        auto& transformer = zeno::FakeTransformer::GetInstance();
        if (!scene->selected.empty() &&
            transformer.isTransformMode() &&
            transformer.clickedAnyHandler(realPos(), dir, front)) {
            transformer.startTransform();
        }
    }
}

void CameraControl::lookTo(int dir) {
    if (dir < 0 || dir > 6) return;
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
    case 6:
        // back to origin
        m_center = {0, 0, 0};
        m_radius = 5.f;
        m_theta = 0.f; m_phi = 0.f;
        Zenovis::GetInstance().updateCameraFront(m_center, -z_axis, y_axis);
    default:
        break;
    }
    m_ortho_mode = true;
    updatePerspective();
    m_ortho_mode = false;
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::clearTransformer() {
    auto& transformer = zeno::FakeTransformer::GetInstance();
    transformer.clear();
}

void CameraControl::changeTransformOperation(const QString& node) {
    auto& transformer = zeno::FakeTransformer::GetInstance();
    auto opt = transformer.getTransOpt();
    transformer.clear();
    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    for (auto const &[key, _] : scene->objectsMan->pairs()) {
        if (key.find(node.toStdString()) != std::string::npos) {
            scene->selected.insert(key);
            transformer.addObject(key);
        }
    }
    transformer.setTransOpt(opt);
    transformer.changeTransOpt();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformOperation(int mode) {
    auto& transformer = zeno::FakeTransformer::GetInstance();
    switch (mode) {
    case 0:
        transformer.toTranslate();
        break;
    case 1:
        transformer.toRotate();
        break;
    case 2:
        transformer.toScale();
        break;
    default:
        break;
    }
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::changeTransformCoordSys() {
    auto& transformer = zeno::FakeTransformer::GetInstance();
    transformer.changeCoordSys();
    zenoApp->getMainWindow()->updateViewport();
}

void CameraControl::resizeTransformHandler(int dir) {
    auto& transformer = zeno::FakeTransformer::GetInstance();
    transformer.resizeHandler(dir);
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
        auto& transformer = zeno::FakeTransformer::GetInstance();
        if (transformer.isTransforming()) {
            auto dir = screenToWorldRay(event->pos().x() / res().x(), event->pos().y() / res().y());
            auto camera_pos = realPos();
            auto x = event->x() * 1.0f;
            auto y = event->y() * 1.0f;
            x = (2 * x / res().x()) - 1;
            y = 1 - (2 * y / res().y());
            auto mouse_pos = glm::vec2(x, y);
            auto vp = scene->camera->m_proj * scene->camera->m_view;
            transformer.transform(camera_pos, mouse_pos, dir, scene->camera->m_lodfront, vp);
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

void CameraControl::fakeMouseDoubleClickEvent(QMouseEvent* event) {
    auto pos = event->pos();
    auto picked_prim = zeno::Picker::GetInstance().just_pick_prim(pos.x(), pos.y());
    if (!picked_prim.empty()) {
        auto obj_node_location = zeno::NodeSyncMgr::GetInstance().searchNodeOfPrim(picked_prim);
        auto subgraph_name = obj_node_location->subgraph.data(ROLE_OBJNAME).toString();
        auto obj_node_name = obj_node_location->node.data(ROLE_OBJID).toString();
        zenoApp->getMainWindow()->editor()->activateTab(subgraph_name, "", obj_node_name);
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
        auto& picker = zeno::Picker::GetInstance();
        auto& transformer = zeno::FakeTransformer::GetInstance();

        if (transformer.isTransforming()) {
            bool moved = false;
            if (m_boundRectStartPos != event->pos()) {
                // create/modify transform primitive node
                moved = true;
            }
            transformer.endTransform(moved);
        }
        else {
            auto cam_pos = realPos();

            scene->select_box = std::nullopt;
            bool shift_pressed = event->modifiers() & Qt::ShiftModifier;
            if (!shift_pressed) {
                scene->selected.clear();
                transformer.clear();
            }

            auto onPrimSelected = []() {
                auto scene = Zenovis::GetInstance().getSession()->get_scene();
                ZenoMainWindow* mainWin = zenoApp->getMainWindow();
                mainWin->onPrimitiveSelected(scene->selected);
            };

            QPoint releasePos = event->pos();
            if (m_boundRectStartPos == releasePos) {
                picker.pick(releasePos.x(), releasePos.y());
                picker.sync_to_scene();
                if (scene->select_mode == zenovis::PICK_OBJECT)
                    onPrimSelected();
                transformer.clear();
                transformer.addObject(picker.get_picked_prims());
            } else {
                int x0 = m_boundRectStartPos.x();
                int y0 = m_boundRectStartPos.y();
                int x1 = releasePos.x();
                int y1 = releasePos.y();
                picker.pick(x0, y0, x1, y1);
                picker.sync_to_scene();
                if (scene->select_mode == zenovis::PICK_OBJECT)
                    onPrimSelected();
                transformer.clear();
                transformer.addObject(picker.get_picked_prims());
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
    , simpleRenderChecked(false)
{
    QGLFormat fmt;
    int nsamples = 16;  // TODO: adjust in a zhouhang-panel
    fmt.setSamples(nsamples);
    fmt.setVersion(3, 2);
    fmt.setProfile(QGLFormat::CoreProfile);
    setFormat(fmt);

    // https://blog.csdn.net/zhujiangm/article/details/90760744
    // https://blog.csdn.net/jays_/article/details/83783871
    setFocusPolicy(Qt::ClickFocus);

    m_camera = std::make_shared<CameraControl>();
    Zenovis::GetInstance().m_camera_control = m_camera.get();

    connect(m_pauseRenderDally, &QTimer::timeout, [&](){
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        scene->drawOptions->simpleRender = false;
        scene->drawOptions->needRefresh = true;
        m_pauseRenderDally->stop();
        //std::cout << "SR: SimpleRender false, Active " << m_pauseRenderDally->isActive() << "\n";
    });
}

void ViewportWidget::setSimpleRenderOption() {
    if(simpleRenderChecked)
        return;

    auto scene = Zenovis::GetInstance().getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
    m_pauseRenderDally->stop();
    m_pauseRenderDally->start(4000);
}

ViewportWidget::~ViewportWidget()
{
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
}

void ViewportWidget::mousePressEvent(QMouseEvent* event)
{
    if(event->button() == Qt::MidButton){
        setSimpleRenderOption();
    }
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* event)
{
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ViewportWidget::wheelEvent(QWheelEvent* event)
{
    setSimpleRenderOption();

    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}

void ViewportWidget::mouseReleaseEvent(QMouseEvent *event) {
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event); 
    update();
}

void ViewportWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseDoubleClickEvent(event);
    update();
}

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

DisplayWidget::DisplayWidget(ZenoMainWindow* pMainWin)
    : QWidget(pMainWin)
    , m_view(nullptr)
    , m_timeline(nullptr)
    , m_mainWin(pMainWin)
    , m_pTimer(nullptr)
    , m_bRecordRun(false)
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

void DisplayWidget::updateFrame(const QString &action) // cihou optix
{
    if (m_mainWin && m_mainWin->inDlgEventLoop())
        return;

    if (action == "newFrame") {
        m_pTimer->stop();
        //zeno::log_warn("stop");
        return;
    } else if (action == "finishFrame") {
        bool bPlaying = Zenovis::GetInstance().isPlaying();
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

void DisplayWidget::runAndRecord(const VideoRecInfo& recInfo)
{
    //reset the record info first.
    m_bRecordRun = true;
    m_recordMgr.setRecordInfo(recInfo);

    Zenovis::GetInstance().startPlay(true);

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

void DisplayWidget::onRecord()
{
    auto& inst = Zenovis::GetInstance();

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
        int frameStart = 0, frameEnd = 0, fps = 0, bitrate = 0, width = 0, height = 0, numOptix = 0, numMSAA = 0;
        QString presets, path, filename;
        bool bRecordOnRun = false;
        dlg.getInfo(frameStart, frameEnd, fps, bitrate, presets, width, height, path, filename, numOptix, numMSAA, bRecordOnRun);
        //validation.

        VideoRecInfo recInfo;
        recInfo.record_path = path;
        recInfo.frameRange = {frameStart, frameEnd};
        recInfo.res = {(float)width, (float)height};
        recInfo.bitrate = bitrate;
        recInfo.fps = fps;
        recInfo.videoname = filename;
        recInfo.numOptix = numOptix;
        recInfo.numMSAA = numMSAA;
        recInfo.bRecordRun = bRecordOnRun;

        m_recordMgr.setRecordInfo(recInfo);

        if (recInfo.bRecordRun)
        {
            connect(&m_recordMgr, &RecordVideoMgr::recordFinished, this, [=]() {
                ZRecordProgressDlg dlgProc(recInfo);
                dlgProc.onRecordFinished();
                dlgProc.exec();
            });
        }
        else
        {
            if (pGlobalComm->maxPlayFrames() == 0)
            {
                QMessageBox::information(nullptr, "Zeno", tr("Run the graph before recording"), QMessageBox::Ok);
                return;
            }

            Zenovis::GetInstance().startPlay(true);

            ZRecordProgressDlg dlgProc(recInfo);
            connect(&m_recordMgr, SIGNAL(frameFinished(int)), &dlgProc, SLOT(onFrameFinished(int)));
            connect(&m_recordMgr, SIGNAL(recordFinished()), &dlgProc, SLOT(onRecordFinished()));
            connect(&m_recordMgr, SIGNAL(recordFailed(QString)), &dlgProc, SLOT(onRecordFailed(QString)));

            dlgProc.show();
            if (QDialog::Accepted == dlgProc.exec())
            {
            }
            else
            {
                m_recordMgr.cancelRecord();
            }
        }
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
        auto scene = Zenovis::GetInstance().getSession()->get_scene();
        auto& picker = zeno::Picker::GetInstance();
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
            picker.set_picked_elems_callback(callback);
            // ----- enter node context
            picker.save_context();
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
                picker.load_from_str(node_context, scene->select_mode);
            }
            picker.sync_to_scene();
            picker.focus(prim_name);
        }
        else {
            picker.load_context();
            picker.sync_to_scene();
            picker.focus("");
            picker.set_picked_elems_callback({});
        }
        zenoApp->getMainWindow()->updateViewport();
    }
}
