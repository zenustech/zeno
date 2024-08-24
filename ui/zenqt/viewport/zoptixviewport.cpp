#include "zoptixviewport.h"
#include "zenovis.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "cameracontrol.h"
#include <zenovis/DrawOptions.h>
#include <zeno/extra/GlobalComm.h>
#include "settings/zenosettingsmanager.h"
#include <zeno/core/Session.h>
#include <zenovis/Camera.h>
#include <zeno/utils/log.h>


ZOptixProcViewport::ZOptixProcViewport(QWidget* parent)
    : QWidget(parent)
    , m_zenovis(nullptr)
    , m_camera(nullptr)
    , updateLightOnce(false)
    , m_bMovingCamera(false)
    , m_worker(nullptr)
{
    m_zenovis = new Zenovis(this);

    setFocusPolicy(Qt::ClickFocus);

    //update timeline
    connect(m_zenovis, &Zenovis::frameUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        //update timeline
        if (mainWin)
            emit mainWin->visFrameUpdated(false, frameid);
    });

    //fake GL
    m_zenovis->initializeGL();
    m_zenovis->setCurrentFrameId(0);    //correct frame automatically.

    m_camera = new CameraControl(m_zenovis, nullptr, nullptr, this);
    m_zenovis->m_camera_control = m_camera;

    const char* e = "optx";
    m_zenovis->getSession()->set_render_engine(e);

    m_worker = new OptixWorker(m_zenovis);

    connect(m_worker, &OptixWorker::renderIterate, this, [=](QImage img) {
        m_renderImage = img;
        update();
    });
    connect(m_worker, &OptixWorker::sig_frameRecordFinished, this, &ZOptixProcViewport::sig_frameRecordFinished);
    connect(m_worker, &OptixWorker::sig_recordFinished, this, &ZOptixProcViewport::sig_recordFinished);

    setRenderSeparately(false, false);
    m_pauseTimer = new QTimer(this);
    m_pauseTimer->stop();
    connect(m_pauseTimer, &QTimer::timeout, this, [=]() {
        m_worker->work();
    });

    m_worker->work();
}

ZOptixProcViewport::~ZOptixProcViewport()
{

}

void ZOptixProcViewport::setSimpleRenderOption()
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
}

void ZOptixProcViewport::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly)
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->updateLightCameraOnly = updateLightCameraOnly;
    scene->drawOptions->updateMatlOnly = updateMatlOnly;
}

void ZOptixProcViewport::cameraLookTo(zenovis::CameraLookToDir dir)
{
    m_camera->lookTo(dir);
}

void ZOptixProcViewport::updateViewport()
{
    m_worker->updateFrame();
}

void ZOptixProcViewport::updateCameraProp(float aperture, float disPlane)
{
    m_camera->setAperture(aperture);
    m_camera->setDisPlane(disPlane);
    m_camera->updatePerspective();
}

void ZOptixProcViewport::updatePerspective()
{
    m_camera->updatePerspective();
}

void ZOptixProcViewport::setCameraRes(const QVector2D& res)
{
    m_camera->setRes(res);
}

void ZOptixProcViewport::setSafeFrames(bool bLock, int nx, int ny)
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->camera->set_safe_frames(bLock, nx, ny);
}

void ZOptixProcViewport::setNumSamples(int samples)
{
    auto scene = m_zenovis->getSession()->get_scene();
    if (scene) {
        scene->drawOptions->num_samples = samples;
    }
}

Zenovis* ZOptixProcViewport::getZenoVis() const
{
    return m_zenovis;
}

bool ZOptixProcViewport::isCameraMoving() const
{
    return m_bMovingCamera;
}

void ZOptixProcViewport::updateCamera()
{
    m_worker->needUpdateCamera();
}

void ZOptixProcViewport::stopRender()
{
    m_worker->stop();
}

void ZOptixProcViewport::resumeRender()
{
    m_worker->work();
}

void ZOptixProcViewport::recordVideo(VideoRecInfo recInfo)
{
    m_worker->recordVideo(recInfo);
}

void ZOptixProcViewport::cancelRecording(VideoRecInfo recInfo)
{
    m_worker->cancelRecording();
}

void ZOptixProcViewport::onMouseHoverMoved()
{
    if (!m_bMovingCamera)
    {
        pauseWorkerAndResume();
    }
}

void ZOptixProcViewport::onFrameSwitched(int frame)
{
    m_worker->onFrameSwitched(frame);
}

void ZOptixProcViewport::onFrameRunFinished(int frame)
{
    emit sig_frameRunFinished(frame);
}

void ZOptixProcViewport::paintEvent(QPaintEvent* event)
{
    if (!m_renderImage.isNull())
    {
        QPainter painter(this);
        painter.drawImage(0, 0, m_renderImage);
    }
}

void ZOptixProcViewport::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    QSize sz = event->size();

    int nx = sz.width();
    int ny = sz.height();

    float ratio = devicePixelRatioF();
    zeno::log_trace("nx={}, ny={}, dpr={}", nx, ny, ratio);
    m_camera->setRes(QVector2D(nx * ratio, ny * ratio));
    m_camera->updatePerspective();
}

void ZOptixProcViewport::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
        setSimpleRenderOption();
        pauseWorkerAndResume();
    }
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ZOptixProcViewport::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = false;
        m_worker->work();
    }
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event);
    update();
}

void ZOptixProcViewport::mouseMoveEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
        pauseWorkerAndResume();
    }
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ZOptixProcViewport::mouseDoubleClickEvent(QMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseDoubleClickEvent(event);
    update();
}

void ZOptixProcViewport::wheelEvent(QWheelEvent* event)
{
    m_bMovingCamera = true;
    pauseWorkerAndResume();
    //m_wheelEventDally->start(100);
    setSimpleRenderOption();

    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
    update();
}

void ZOptixProcViewport::keyPressEvent(QKeyEvent* event)
{
    _base::keyPressEvent(event);
    //qInfo() << event->key();
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    int key = settings.getShortCut(ShortCut_MovingHandler);
    int uKey = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    if (modifiers & Qt::ShiftModifier) {
        uKey += Qt::SHIFT;
    }
    if (modifiers & Qt::ControlModifier) {
        uKey += Qt::CTRL;
    }
    if (modifiers & Qt::AltModifier) {
        uKey += Qt::ALT;
    }
    /*
    if (uKey == key)
        this->changeTransformOperation(0);
    key = settings.getShortCut(ShortCut_RevolvingHandler);
    if (uKey == key)
        this->changeTransformOperation(1);
    key = settings.getShortCut(ShortCut_ScalingHandler);
    if (uKey == key)
        this->changeTransformOperation(2);
    key = settings.getShortCut(ShortCut_CoordSys);
    if (uKey == key)
        this->changeTransformCoordSys();
    */

    key = settings.getShortCut(ShortCut_FrontView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::front_view);
    key = settings.getShortCut(ShortCut_RightView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::right_view);
    key = settings.getShortCut(ShortCut_VerticalView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::top_view);
    key = settings.getShortCut(ShortCut_InitViewPos);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::back_to_origin);

    key = settings.getShortCut(ShortCut_BackView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::back_view);
    key = settings.getShortCut(ShortCut_LeftView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::left_view);
    key = settings.getShortCut(ShortCut_UpwardView);
    if (uKey == key)
        this->cameraLookTo(zenovis::CameraLookToDir::bottom_view);

    key = settings.getShortCut(ShortCut_InitHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(0);
    key = settings.getShortCut(ShortCut_AmplifyHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(1);
    key = settings.getShortCut(ShortCut_ReduceHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(2);
}

void ZOptixProcViewport::keyReleaseEvent(QKeyEvent* event)
{
    _base::keyReleaseEvent(event);
}

void ZOptixProcViewport::pauseWorkerAndResume()
{
    m_worker->stop();
    m_pauseTimer->start(m_resumeTime);
}

void ZOptixProcViewport::screenshoot(QString path, QString type, int resx, int resy)
{
    //todo
}

void ZOptixProcViewport::setSlidFeq(int feq)
{
    //todo
}