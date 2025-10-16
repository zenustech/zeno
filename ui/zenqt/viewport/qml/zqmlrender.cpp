#include "zqmlrender.h"
#include "objloader.h"
#include <QMatrix4x4>
#include <QDebug>
#include <cmath>
#include <QtMath>
#include "viewport/zenovis.h"
#include "viewport/cameracontrol.h"
#include "model/graphsmanager.h"
#include <zenovis/DrawOptions.h>
#include "zenoapplication.h"


namespace {
    struct OpenGLProcAddressHelper {
        inline static QOpenGLContext* ctx;

        static void* getProcAddress(const char* name) {
            return (void*)ctx->getProcAddress(name);
        }
    };
}


ZQmlRender::ZQmlRender(QSize res, QObject* parent)
    : QObject(parent)
    , m_camera(nullptr)
{
    m_zenovis = new Zenovis(this);
    m_camera = new CameraControl(m_zenovis, m_fakeTrans, m_picker, this);
    m_zenovis->m_camera_control = m_camera;
    m_zenovis->initializeGL();  //actually, this is a func init session.
    resize(res.width(), res.height());
}

ZQmlRender::~ZQmlRender()
{
}

void ZQmlRender::reload_objects(const zeno::render_reload_info& info)
{
    m_zenovis->reload(info);
    emit requestUpdate();
    emit sig_reload_finished();
}

void ZQmlRender::onMouseEvent(ViewMouseInfo event_info)
{
    if (event_info.type != QEvent::None) {
        if (QEvent::MouseButtonPress == event_info.type) {
            fakeMousePressEvent(event_info);
        }
        else if (QEvent::MouseButtonRelease == event_info.type) {
            fakeMouseReleaseEvent(event_info);
        }
        else if (QEvent::MouseMove == event_info.type) {
            fakeMouseMoveEvent(event_info);
        }
        else if (QEvent::Wheel == event_info.type) {
            fakeWheelEvent(event_info);
        }
    }
}

void ZQmlRender::initialize(QOpenGLContext* context, CoordinateMirroring cm)
{
    OpenGLProcAddressHelper::ctx = context;
    assert(OpenGLProcAddressHelper::ctx);
    m_zenovis->loadGLAPI((void*)OpenGLProcAddressHelper::getProcAddress);

    //计算端可能已经运行了一波结果，这里需要先同步掉已有的计算结果
    zeno::render_reload_info info;
    GraphsManager* graphsMgr = zenoApp->graphsManager();
    info.current_ui_graph = graphsMgr->currentGraphPath().toStdString();
    info.policy = zeno::Reload_SwitchGraph;
    if (graphsMgr->getGraph({ "main" }) && info.current_ui_graph != "") {
        reload_objects(info);
    }
}

void ZQmlRender::render()
{
    m_zenovis->paintGL();
}

Zenovis* ZQmlRender::getZenovis() const {
    return m_zenovis;
}

void ZQmlRender::fakeMousePressEvent(ViewMouseInfo event_info) {
    m_camera->fakeMousePressEvent(event_info);
}

void ZQmlRender::fakeMouseReleaseEvent(ViewMouseInfo event_info) {
    m_camera->fakeMouseReleaseEvent(event_info);
}

void ZQmlRender::fakeMouseMoveEvent(ViewMouseInfo event_info) {
    m_camera->fakeMouseMoveEvent(event_info);
}

void ZQmlRender::fakeWheelEvent(ViewMouseInfo event_info) {
    m_camera->fakeWheelEvent(event_info);
}

void ZQmlRender::resize(int nx, int ny)
{
    if (m_camera) {
        m_camera->setRes(QVector2D(nx, ny));
        m_camera->updatePerspective();
        emit requestUpdate();
    }
}

void ZQmlRender::updatePerspective()
{
    m_camera->updatePerspective();
}

void ZQmlRender::setNumSamples(int samples) {
    auto scene = getSession()->get_scene();
    if (scene) {
        scene->drawOptions->num_samples = samples;
    }
}

void ZQmlRender::setCameraRes(const QVector2D& res) {
    m_camera->setRes(res);
}

void ZQmlRender::setShowptnum(bool bShow) {
    m_zenovis->getSession()->set_show_ptnum(bShow);
}

void ZQmlRender::cleanUpScene() {
    m_zenovis->cleanUpScene();
    emit requestUpdate();
}

zenovis::Session* ZQmlRender::getSession() const
{
    return m_zenovis->getSession();
}

void ZQmlRender::setSafeFrames(bool bLock, int nx, int ny) {
    auto scene = m_zenovis->getSession()->get_scene();
    scene->camera->set_safe_frames(bLock, nx, ny);
}

void ZQmlRender::setSimpleRenderOption()
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
    //m_pauseRenderDally->stop();
    //m_pauseRenderDally->start(simpleRenderTime * 1000);  // Second to millisecond
}

void ZQmlRender::clearTransformer()
{
    m_camera->clearTransformer();
}

void ZQmlRender::changeTransformOperationByNode(const QString& node)
{
    m_camera->changeTransformOperation(node);
}

void ZQmlRender::changeTransformOperation(int mode)
{
    m_camera->changeTransformOperation(mode);
}

void ZQmlRender::cameraLookTo(zenovis::CameraLookToDir dir)
{
    m_camera->lookTo(dir);
}

void ZQmlRender::setViewWidgetInfo(DockContentWidgetInfo& info)
{

}
