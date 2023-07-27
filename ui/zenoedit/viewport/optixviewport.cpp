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


OptixWorker::OptixWorker(Zenovis *pzenoVis)
    : QObject(nullptr)
    , m_zenoVis(pzenoVis)
    , m_bRecording(false)
{
    m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

OptixWorker::~OptixWorker()
{
}

OptixWorker::OptixWorker(QObject* parent)
    : QObject(parent)
    , m_zenoVis(nullptr)
    , m_pTimer(nullptr)
    , m_bRecording(false)
{
    //used by offline worker.
    m_pTimer = new QTimer(this);
    m_zenoVis = new Zenovis(this);

    //fake GL
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

void OptixWorker::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly) {
    auto scene = m_zenoVis->getSession()->get_scene();
    scene->drawOptions->updateLightCameraOnly = updateLightCameraOnly;
    scene->drawOptions->updateMatlOnly = updateMatlOnly;
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

void OptixWorker::onSetLoopPlaying(bool enbale)
{
    m_zenoVis->setLoopPlaying(enbale);
}

void OptixWorker::stop()
{
    m_pTimer->stop();
    //todo: use a flag to mark, otherwise the timer will be resumed.
}

void OptixWorker::work()
{
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
    , m_zenovis(nullptr)
    , m_camera(nullptr)
    , updateLightOnce(false)
    , m_bMovingCamera(false)
{
    m_zenovis = new Zenovis(this);

    setFocusPolicy(Qt::ClickFocus);

    connect(m_zenovis, &Zenovis::objectsUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        //if (mainWin)
        //    emit mainWin->visObjectsUpdated(this, frameid);
    });

    //no need to notify timeline to update.
    /*
    connect(m_zenovis, &Zenovis::frameUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin)
            emit mainWin->visFrameUpdated(false, frameid);
    }, Qt::BlockingQueuedConnection);
    */

    //fake GL
    m_zenovis->initializeGL();
    m_zenovis->setCurrentFrameId(0);    //correct frame automatically.

    m_camera = new CameraControl(m_zenovis, nullptr, nullptr, this);
    m_zenovis->m_camera_control = m_camera;

    const char *e = "optx";
    m_zenovis->getSession()->set_render_engine(e);

    auto scene = m_zenovis->getSession()->get_scene();

    m_worker = new OptixWorker(m_zenovis);
    m_worker->moveToThread(&m_thdOptix);
    connect(&m_thdOptix, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(&m_thdOptix, &QThread::started, m_worker, &OptixWorker::work);
    connect(m_worker, &OptixWorker::renderIterate, this, [=](QImage img) {
        m_renderImage = img;
        update();
    });
    connect(this, &ZOptixViewport::cameraAboutToRefresh, m_worker, &OptixWorker::needUpdateCamera);
    connect(this, &ZOptixViewport::stopRenderOptix, m_worker, &OptixWorker::stop);
    connect(this, &ZOptixViewport::resumeWork, m_worker, &OptixWorker::work);
    connect(this, &ZOptixViewport::sigRecordVideo, m_worker, &OptixWorker::recordVideo, Qt::QueuedConnection);
    connect(this, &ZOptixViewport::sig_setSafeFrames, m_worker, &OptixWorker::onSetSafeFrames);

    connect(m_worker, &OptixWorker::sig_recordFinished, this, &ZOptixViewport::sig_recordFinished);
    connect(m_worker, &OptixWorker::sig_frameRecordFinished, this, &ZOptixViewport::sig_frameRecordFinished);

    connect(this, &ZOptixViewport::sig_switchTimeFrame, m_worker, &OptixWorker::onFrameSwitched);
    connect(this, &ZOptixViewport::sig_togglePlayButton, m_worker, &OptixWorker::onPlayToggled);
    connect(this, &ZOptixViewport::sig_setRenderSeparately, m_worker, &OptixWorker::setRenderSeparately);
    connect(this, &ZOptixViewport::sig_setLoopPlaying, m_worker, &OptixWorker::onSetLoopPlaying);

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
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
}

void ZOptixViewport::setRenderSeparately(bool updateLightCameraOnly, bool updateMatlOnly) {
    emit sig_setRenderSeparately(updateLightCameraOnly, updateMatlOnly);
}

void ZOptixViewport::cameraLookTo(int dir)
{
    m_camera->lookTo(dir);
}

Zenovis* ZOptixViewport::getZenoVis() const
{
    return m_zenovis;
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
    m_worker->cancelRecording();
}

void ZOptixViewport::onFrameRunFinished(int frame)
{
    emit sig_frameRunFinished(frame);
}

void ZOptixViewport::updateCameraProp(float aperture, float disPlane)
{
    m_camera->setAperture(aperture);
    m_camera->setDisPlane(disPlane);
    m_camera->updatePerspective();
}

void ZOptixViewport::updatePerspective()
{
    m_camera->updatePerspective();
}

void ZOptixViewport::setCameraRes(const QVector2D& res)
{
    m_camera->setRes(res);
}

void ZOptixViewport::setSafeFrames(bool bLock, int nx, int ny)
{
    emit sig_setSafeFrames(bLock, nx, ny);
}

void ZOptixViewport::setNumSamples(int samples)
{
    auto scene = m_zenovis->getSession()->get_scene();
    if (scene) {
        scene->drawOptions->num_samples = samples;
    }
}

void ZOptixViewport::resizeEvent(QResizeEvent* event)
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

void ZOptixViewport::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
        setSimpleRenderOption();
    }
    _base::mousePressEvent(event);
    m_camera->fakeMousePressEvent(event);
    update();
}

void ZOptixViewport::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = false;
    }
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseReleaseEvent(event);
    update();
}

void ZOptixViewport::mouseMoveEvent(QMouseEvent* event)
{
    if (event->button() == Qt::MidButton) {
        m_bMovingCamera = true;
    }
    setSimpleRenderOption();

    _base::mouseMoveEvent(event);
    m_camera->fakeMouseMoveEvent(event);
    update();
}

void ZOptixViewport::mouseDoubleClickEvent(QMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    m_camera->fakeMouseDoubleClickEvent(event);
    update();
}

void ZOptixViewport::wheelEvent(QWheelEvent* event)
{
    m_bMovingCamera = true;
    //m_wheelEventDally->start(100);
    setSimpleRenderOption();

    _base::wheelEvent(event);
    m_camera->fakeWheelEvent(event);
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
        m_camera->resizeTransformHandler(0);
    key = settings.getShortCut(ShortCut_AmplifyHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(1);
    key = settings.getShortCut(ShortCut_ReduceHandler);
    if (uKey == key)
        m_camera->resizeTransformHandler(2);
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