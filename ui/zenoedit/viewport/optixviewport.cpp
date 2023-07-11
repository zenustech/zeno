#include "optixviewport.h"
#include "zenovis.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "cameracontrol.h"
#include <zenovis/DrawOptions.h>
#include <zeno/extra/GlobalComm.h>
#include "settings/zenosettingsmanager.h"
#include "launch/corelaunch.h"
#include <zeno/core/Session.h>
#include <zenovis/Camera.h>


OptixWorker::~OptixWorker()
{
}

OptixWorker::OptixWorker(QObject* parent)
    : QObject(parent)
    , m_zenoVis(nullptr)
    , m_pTimer(nullptr)
    , m_bRecording(false)
    , m_camera(nullptr)
{
    //used by offline worker.
    initialize();

    m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

void OptixWorker::initialize()
{
    m_zenoVis = new Zenovis(this);
    m_zenoVis->initializeGL();
    m_zenoVis->setCurrentFrameId(0);    //correct frame automatically.
    m_zenoVis->m_camera_control = new CameraControl(m_zenoVis, nullptr, nullptr, this);
    m_zenoVis->getSession()->set_render_engine("optx");
}

void OptixWorker::updateFrame()
{
    //avoid conflict.
    if (m_bRecording)
        return;

    m_zenoVis->paintGL();
    int w = 0, h = 0;
    void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);

    m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
    m_renderImg = m_renderImg.mirrored(false, true);

    emit renderIterate(m_renderImg);
}

void OptixWorker::onPlayToggled(bool bToggled)
{
    //todo: priority.
    m_zenoVis->startPlay(bToggled);
    m_pTimer->start(16);
}

void OptixWorker::onFrameSwitched(int frame)
{
    //ui switch.
    m_zenoVis->setCurrentFrameId(frame);
    m_zenoVis->startPlay(false);
}

void OptixWorker::cancelRecording()
{
    m_bRecording = false;
}

void OptixWorker::setResolution(const QVector2D& res)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->setRes(res);
    pCamera->updatePerspective();
}

void OptixWorker::setSimpleRenderOption()
{
    auto scene = m_zenoVis->getSession()->get_scene();
    ZASSERT_EXIT(scene);
    scene->drawOptions->simpleRender = true;
}

void OptixWorker::cameraLookTo(int dir)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->lookTo(dir);
}

void OptixWorker::updateCameraProp(float aperture, float disPlane)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->setAperture(aperture);
    pCamera->setDisPlane(disPlane);
    pCamera->updatePerspective();
}

void OptixWorker::resizeTransformHandler(int dir)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->resizeTransformHandler(dir);
}

void OptixWorker::fakeMousePressEvent(QMouseEvent* event)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->fakeMousePressEvent(event);
}

void OptixWorker::fakeMouseReleaseEvent(QMouseEvent* event)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->fakeMouseReleaseEvent(event);
}

void OptixWorker::fakeMouseMoveEvent(QMouseEvent* event)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->fakeMouseMoveEvent(event);
}

void OptixWorker::fakeWheelEvent(QWheelEvent* event)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->fakeWheelEvent(event);
}

void OptixWorker::fakeMouseDoubleClickEvent(QMouseEvent* event)
{
    CameraControl* pCamera = m_zenoVis->getCameraControl();
    ZASSERT_EXIT(pCamera);
    pCamera->fakeMouseDoubleClickEvent(event);
}

void OptixWorker::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly) {
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->drawOptions->updateLightCameraOnly = updateLightCameraOnly;
    scene->drawOptions->updateMatlOnly = updateMatlOnly;
}

void OptixWorker::setNumSamples(int samples)
{
    auto scene = m_zenoVis->getSession()->get_scene();
    if (scene) {
        scene->drawOptions->num_samples = samples;
    }
}

void OptixWorker::onSetSafeFrames(bool bLock, int nx, int ny) {
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->camera->set_safe_frames(bLock, nx, ny);
}

void OptixWorker::recordVideo(VideoRecInfo recInfo)
{
    //for the case about recording after run.
    zeno::scope_exit sp([=] {
        m_bRecording = false;
        m_pTimer->start(16);
    });

    m_bRecording = true;
    m_pTimer->stop();

    for (int frame = recInfo.frameRange.first; frame <= recInfo.frameRange.second;)
    {
        if (!m_bRecording)
        {
            emit sig_recordCanceled();
            return;
        }
        bool bSucceed = recordFrame_impl(recInfo, frame);
        if (bSucceed)
        {
            frame++;
        }
        else
        {
            QThread::sleep(0);
        }
    }
    emit sig_recordFinished();
}

bool OptixWorker::recordFrame_impl(VideoRecInfo recInfo, int frame)
{
    auto record_file = zeno::format("{}/P/{:07d}.jpg", recInfo.record_path.toStdString(), frame);
    auto extname = QFileInfo(QString::fromStdString(record_file)).suffix().toStdString();

    auto scene = m_zenoVis->getSession()->get_scene();
    auto old_num_samples = scene->drawOptions->num_samples;
    scene->drawOptions->num_samples = recInfo.numOptix;
    scene->drawOptions->denoise = recInfo.needDenoise;

    zeno::scope_exit sp([=]() {scene->drawOptions->num_samples = old_num_samples;});
    //it seems that msaa is used by opengl, but opengl has been removed from optix.
    scene->drawOptions->msaa_samples = recInfo.numMSAA;

    auto [x, y] = m_zenoVis->getSession()->get_window_size();

    auto &globalComm = zeno::getSession().globalComm;
    int numOfFrames = globalComm->numOfFinishedFrame();
    if (numOfFrames == 0)
        return false;

    std::pair<int, int> frameRg = globalComm->frameRange();
    int beginFrame = frameRg.first;
    int endFrame = frameRg.first + numOfFrames - 1;
    if (frame < beginFrame || frame > endFrame)
        return false;

    int actualFrame = m_zenoVis->setCurrentFrameId(frame);
    m_zenoVis->doFrameUpdate();
    //todo: may be the frame has not been finished, in this case, we have to wait.

    m_zenoVis->getSession()->set_window_size((int)recInfo.res.x(), (int)recInfo.res.y());
    m_zenoVis->getSession()->do_screenshot(record_file, extname);
    m_zenoVis->getSession()->set_window_size(x, y);

    //todo: emit some signal to main thread(ui)
    emit sig_frameRecordFinished(frame);

    if (1) {
        //update ui.
        int w = 0, h = 0;
        void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);
        m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
        m_renderImg = m_renderImg.mirrored(false, true);
        emit renderIterate(m_renderImg);
    }
    return true;
}

void OptixWorker::stop()
{
    m_pTimer->stop();
    //todo: use a flag to mark, otherwise the timer will be resumed.
}

void OptixWorker::onWorkThreadStarted()
{
    initialize();
    m_pTimer->start(16);
}

QImage OptixWorker::renderImage() const
{
    return m_renderImg;
}

void OptixWorker::needUpdateCamera()
{
    //todo: update reason.
    //m_zenoVis->getSession()->get_scene()->drawOptions->needUpdateGeo = false;	//just for teset.
    m_zenoVis->getSession()->get_scene()->drawOptions->needRefresh = true;
    m_pTimer->start(16);
}


ZOptixViewport::ZOptixViewport(QWidget* parent)
    : QWidget(parent)
    , updateLightOnce(false)
    , m_bMovingCamera(false)
{
    setFocusPolicy(Qt::ClickFocus);

    OptixWorker* pWorker = new OptixWorker;
    pWorker->moveToThread(&m_thdOptix);
    connect(&m_thdOptix, &QThread::finished, pWorker, &QObject::deleteLater);
    connect(&m_thdOptix, &QThread::started, pWorker, &OptixWorker::onWorkThreadStarted);
    connect(pWorker, &OptixWorker::renderIterate, this, [=](QImage img) {
        m_renderImage = img;
        update();
    });
    connect(this, &ZOptixViewport::cameraAboutToRefresh, pWorker, &OptixWorker::needUpdateCamera);
    connect(this, &ZOptixViewport::stopRenderOptix, pWorker, &OptixWorker::stop);
    connect(this, &ZOptixViewport::resumeWork, pWorker, &OptixWorker::onWorkThreadStarted);
    connect(this, &ZOptixViewport::sigRecordVideo, pWorker, &OptixWorker::recordVideo, Qt::QueuedConnection);
    connect(this, &ZOptixViewport::sig_setSafeFrames, pWorker, &OptixWorker::onSetSafeFrames);

    connect(pWorker, &OptixWorker::sig_recordFinished, this, &ZOptixViewport::sig_recordFinished);
    connect(pWorker, &OptixWorker::sig_frameRecordFinished, this, &ZOptixViewport::sig_frameRecordFinished);

    connect(this, &ZOptixViewport::sig_switchTimeFrame, pWorker, &OptixWorker::onFrameSwitched);
    connect(this, &ZOptixViewport::sig_togglePlayButton, pWorker, &OptixWorker::onPlayToggled);
    connect(this, &ZOptixViewport::sig_setRenderSeparately, pWorker, &OptixWorker::setRenderSeparately);
    connect(this, &ZOptixViewport::sig_setResolution, pWorker, &OptixWorker::setResolution);
    connect(this, &ZOptixViewport::sig_cancelRecording, pWorker, &OptixWorker::cancelRecording);
    connect(this, &ZOptixViewport::sig_setSimpleRenderOption, pWorker, &OptixWorker::setSimpleRenderOption);
    connect(this, &ZOptixViewport::sig_cameraLookTo, pWorker, &OptixWorker::cameraLookTo);
    connect(this, &ZOptixViewport::sig_updateCameraProp, pWorker, &OptixWorker::updateCameraProp);
    connect(this, &ZOptixViewport::sig_resizeTransformHandler, pWorker, &OptixWorker::resizeTransformHandler);
    connect(this, &ZOptixViewport::sig_setNumSamples, pWorker, &OptixWorker::setNumSamples);
    //unknown event .
    connect(this, &ZOptixViewport::sig_fakeMousePressEvent, pWorker, &OptixWorker::fakeMousePressEvent);
    connect(this, &ZOptixViewport::sig_fakeMouseReleaseEvent, pWorker, &OptixWorker::fakeMouseReleaseEvent);
    connect(this, &ZOptixViewport::sig_fakeMouseMoveEvent, pWorker, &OptixWorker::fakeMouseMoveEvent);
    connect(this, &ZOptixViewport::sig_fakeWheelEvent, pWorker, &OptixWorker::fakeWheelEvent);
    connect(this, &ZOptixViewport::sig_fakeMouseDoubleClickEvent, pWorker, &OptixWorker::fakeMouseDoubleClickEvent);

    setRenderSeparately(false, false);
    m_thdOptix.start();
}

ZOptixViewport::~ZOptixViewport()
{
    m_thdOptix.quit();
    m_thdOptix.wait();
}

void ZOptixViewport::setSimpleRenderOption()
{
    emit sig_setSimpleRenderOption();
}

void ZOptixViewport::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly) {
    emit sig_setRenderSeparately(updateLightCameraOnly, updateMatlOnly);
}

void ZOptixViewport::cameraLookTo(int dir)
{
    emit sig_cameraLookTo(dir);
}

Zenovis* ZOptixViewport::getZenoVis() const
{
    return nullptr;
}

bool ZOptixViewport::isCameraMoving() const
{
    return m_bMovingCamera;
}

void ZOptixViewport::updateCamera()
{
    emit cameraAboutToRefresh();
}

void ZOptixViewport::killThread()
{
    stopRender();
    m_thdOptix.quit();
    m_thdOptix.wait();
}

void ZOptixViewport::stopRender()
{
    emit stopRenderOptix();
}

void ZOptixViewport::resumeRender()
{
    emit resumeWork();
}

void ZOptixViewport::recordVideo(VideoRecInfo recInfo)
{
    emit sigRecordVideo(recInfo);
}

void ZOptixViewport::cancelRecording(VideoRecInfo recInfo)
{
    emit sig_cancelRecording();
}

void ZOptixViewport::onFrameRunFinished(int frame)
{
    emit sig_frameRunFinished(frame);
}

void ZOptixViewport::updateCameraProp(float aperture, float disPlane)
{
    emit sig_updateCameraProp(aperture, disPlane);
}

void ZOptixViewport::updatePerspective()
{
    //only solidRunRender refer this, and this is the case for glviewport.
}

void ZOptixViewport::setCameraRes(const QVector2D& res)
{
    emit sig_setResolution(res);
}

void ZOptixViewport::setSafeFrames(bool bLock, int nx, int ny)
{
    emit sig_setSafeFrames(bLock, nx, ny);
}

void ZOptixViewport::setNumSamples(int samples)
{
    emit sig_setNumSamples(samples);
}

void ZOptixViewport::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    QSize sz = event->size();

    int nx = sz.width();
    int ny = sz.height();

    float ratio = devicePixelRatioF();
    zeno::log_trace("nx={}, ny={}, dpr={}", nx, ny, ratio);
    emit sig_setResolution(QVector2D(nx * ratio, ny * ratio));
}

void ZOptixViewport::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
        setSimpleRenderOption();
    }
    _base::mousePressEvent(event);
    emit sig_fakeMousePressEvent(event);
    update();
}

void ZOptixViewport::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = false;
    }
    _base::mouseReleaseEvent(event);
    emit sig_fakeMouseReleaseEvent(event);
    update();
}

void ZOptixViewport::mouseMoveEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
    }
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    emit sig_fakeMouseMoveEvent(event);
    update();
}

void ZOptixViewport::mouseDoubleClickEvent(QMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    emit sig_fakeMouseDoubleClickEvent(event);
    update();
}

void ZOptixViewport::wheelEvent(QWheelEvent* event)
{
    m_bMovingCamera = true;
    //m_wheelEventDally->start(100);
    setSimpleRenderOption();

    _base::wheelEvent(event);
    emit sig_fakeWheelEvent(event);
    update();
}

void ZOptixViewport::keyPressEvent(QKeyEvent* event)
{
    _base::keyPressEvent(event);
    //qInfo() << event->key();
    ZenoSettingsManager &settings = ZenoSettingsManager::GetInstance();
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
        this->cameraLookTo(0);
    key = settings.getShortCut(ShortCut_RightView);
    if (uKey == key)
        this->cameraLookTo(1);
    key = settings.getShortCut(ShortCut_VerticalView);
    if (uKey == key)
        this->cameraLookTo(2);
    key = settings.getShortCut(ShortCut_InitViewPos);
    if (uKey == key)
        this->cameraLookTo(6);

    key = settings.getShortCut(ShortCut_BackView);
    if (uKey == key)
        this->cameraLookTo(3);
    key = settings.getShortCut(ShortCut_LeftView);
    if (uKey == key)
        this->cameraLookTo(4);
    key = settings.getShortCut(ShortCut_UpwardView);
    if (uKey == key)
        this->cameraLookTo(5);

    key = settings.getShortCut(ShortCut_InitHandler);
    if (uKey == key)
        emit sig_resizeTransformHandler(0);
    key = settings.getShortCut(ShortCut_AmplifyHandler);
    if (uKey == key)
        emit sig_resizeTransformHandler(1);
    key = settings.getShortCut(ShortCut_ReduceHandler);
    if (uKey == key)
        emit sig_resizeTransformHandler(2);
}

void ZOptixViewport::keyReleaseEvent(QKeyEvent* event)
{
    _base::keyReleaseEvent(event);
}

void ZOptixViewport::paintEvent(QPaintEvent* event)
{
    if (!m_renderImage.isNull())
    {
        QPainter painter(this);
        painter.drawImage(0, 0, m_renderImage);
    }
}