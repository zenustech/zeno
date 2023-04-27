#include "optixviewport.h"
#include "zenovis.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "cameracontrol.h"
#include <zenovis/DrawOptions.h>
#include "settings/zenosettingsmanager.h"



OptixWorker::OptixWorker(Zenovis *pzenoVis)
    : QObject(nullptr)
    , m_zenoVis(pzenoVis)
{
    m_pTimer = new QTimer(this);
    connect(m_pTimer, SIGNAL(timeout()), this, SLOT(updateFrame()));
}

void OptixWorker::updateFrame()
{
    m_zenoVis->paintGL();
    int w, h;
    void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);

    m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
    m_renderImg = m_renderImg.mirrored(false, true);
    emit renderIterate(m_renderImg);
}

void OptixWorker::work()
{
    m_pTimer->start(16);
    /*
    while (true)
    {
        m_zenoVis->paintGL();
        int w, h;
        void *data = m_zenoVis->getSession()->get_scene()->getOptixImg(w, h);

        m_renderImg = QImage((uchar *)data, w, h, QImage::Format_RGBA8888);
        m_renderImg = m_renderImg.mirrored(false, true);
        emit renderIterate(m_renderImg);
    }
    */
}

QImage OptixWorker::renderImage() const
{
    return m_renderImg;
}

void OptixWorker::needUpdateCamera()
{
    //todo: update reason.
    m_zenoVis->getSession()->get_scene()->drawOptions->needRefresh = true;
    m_pTimer->start();
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

    connect(m_zenovis, &Zenovis::frameUpdated, this, [=](int frameid) {
        auto mainWin = zenoApp->getMainWindow();
        if (mainWin)
            mainWin->visFrameUpdated(frameid);
    });

    //fake GL
    m_zenovis->initializeGL();

    m_camera = new CameraControl(m_zenovis, nullptr, nullptr, this);
    m_zenovis->m_camera_control = m_camera;

    const char *e = "optx";
    m_zenovis->getSession()->set_render_engine(e);

    auto scene = m_zenovis->getSession()->get_scene();

    OptixWorker *worker = new OptixWorker(m_zenovis);
    worker->moveToThread(&m_thdOptix);
    connect(&m_thdOptix, &QThread::finished, worker, &QObject::deleteLater);
    connect(&m_thdOptix, &QThread::started, worker, &OptixWorker::work);
    connect(worker, &OptixWorker::renderIterate, this, [=](QImage img) {
        m_renderImage = img;
        update();
    });
    connect(this, &ZOptixViewport::cameraAboutToRefresh, worker, &OptixWorker::needUpdateCamera);

    m_thdOptix.start();
}

ZOptixViewport::~ZOptixViewport()
{
}

void ZOptixViewport::setSimpleRenderOption()
{
    auto scene = m_zenovis->getSession()->get_scene();
    scene->drawOptions->simpleRender = true;
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

void ZOptixViewport::updateCameraProp(float aperture, float disPlane)
{
    m_camera->setAperture(aperture);
    m_camera->setDisPlane(disPlane);
    m_camera->updatePerspective();
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

    m_zenovis->getSession()->set_window_size(sz.width(), sz.height());
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