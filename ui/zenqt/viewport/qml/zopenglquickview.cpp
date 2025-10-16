#include "zopenglquickview.h"
#include "camera.h"
#include "zqmlrender.h"
#include <QSurfaceFormat>
#include <QQmlContext>
#include "settings/zenosettingsmanager.h"


ZOpenGLQuickView::ZOpenGLQuickView(QWindow* parent)
    : QQuickView(parent)
    , m_renderer(nullptr)
{
    QSurfaceFormat format;
    format.setMajorVersion(3);
    format.setMinorVersion(3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4);
    setFormat(format);

    connect(this, &QQuickWindow::sceneGraphInitialized,
        this, &ZOpenGLQuickView::initializeUnderlay,
        Qt::DirectConnection);

    connect(this, &QQuickWindow::beforeSynchronizing,
        this, &ZOpenGLQuickView::synchronizeUnderlay,
        Qt::DirectConnection);

    connect(this, &QQuickWindow::beforeRendering,
        this, &ZOpenGLQuickView::renderUnderlay,
        Qt::DirectConnection);

    connect(this, &QQuickWindow::sceneGraphInvalidated,
        this, &ZOpenGLQuickView::invalidateUnderlay,
        Qt::DirectConnection);

    setClearBeforeRendering(false);
    setPersistentOpenGLContext(true);
    setResizeMode(SizeRootObjectToView);
}

void ZOpenGLQuickView::initializeUnderlay() {
    //QSGRenderThread
    auto ctx = this->openglContext();
    QSize sz = this->size();
    const qreal scaleFactor = devicePixelRatio();
    m_renderer.reset(new ZQmlRender(sz * scaleFactor, this));
    m_renderer->initialize(ctx);
    resetOpenGLState();

    connect(this, &ZOpenGLQuickView::sig_Reload, m_renderer.get(),
        &ZQmlRender::reload_objects, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_MouseEvent, m_renderer.get(),
        &ZQmlRender::onMouseEvent, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_Resize, m_renderer.get(),
        &ZQmlRender::resize, Qt::QueuedConnection);

    connect(this, &ZOpenGLQuickView::sig_UpdatePerspective, m_renderer.get(), &ZQmlRender::updatePerspective, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_SetNumSamples, m_renderer.get(), &ZQmlRender::setNumSamples, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_setCameraRes, m_renderer.get(), &ZQmlRender::setCameraRes, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_setSafeFrames, m_renderer.get(), &ZQmlRender::setSafeFrames, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_setSimpleRenderOption, m_renderer.get(), &ZQmlRender::setSimpleRenderOption, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_clearTransformer, m_renderer.get(), &ZQmlRender::clearTransformer, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_cameraLookTo, m_renderer.get(), &ZQmlRender::cameraLookTo, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_changeTransformOperationByNode, m_renderer.get(), &ZQmlRender::changeTransformOperationByNode, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_changeTransformOperation, m_renderer.get(), &ZQmlRender::changeTransformOperation, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_cleanScene, m_renderer.get(), &ZQmlRender::cleanUpScene, Qt::QueuedConnection);
    connect(this, &ZOpenGLQuickView::sig_showptnum, m_renderer.get(), &ZQmlRender::setShowptnum, Qt::QueuedConnection);

    connect(m_renderer.get(), &ZQmlRender::requestUpdate, this,
        &ZOpenGLQuickView::onUpdateRequest, Qt::QueuedConnection);
    connect(m_renderer.get(), &ZQmlRender::sig_reload_finished, this,
        &ZOpenGLQuickView::sig_render_reload_finished);
}

void ZOpenGLQuickView::onUpdateRequest()
{
    update();
}

void ZOpenGLQuickView::synchronizeUnderlay() {
#ifdef BASE_KDAB
    m_renderer->setAzimuth(m_camera->azimuth());
    m_renderer->setElevation(m_camera->elevation());
    m_renderer->setDistance(m_camera->distance());
#endif
}

void ZOpenGLQuickView::reload_objects(const zeno::render_reload_info& info) {
    emit sig_Reload(info);  //connect to: ZQmlRender::reload_objects
}

void ZOpenGLQuickView::renderUnderlay() {
    if (m_renderer) {
        m_renderer->render();
        resetOpenGLState();
    }
}

void ZOpenGLQuickView::invalidateUnderlay() {
    if (m_renderer) {
        resetOpenGLState();
    }
}

void ZOpenGLQuickView::mousePressEvent(QMouseEvent* event)
{
    int button = Qt::NoButton;
    ZenoSettingsManager& settings = ZenoSettingsManager::GetInstance();
    settings.getViewShortCut(ShortCut_MovingView, button);
    settings.getViewShortCut(ShortCut_RotatingView, button);
    if (event->button() & button) {
        //m_bMovingCamera = true;
        //setSimpleRenderOption();
    }
    QQuickView::mousePressEvent(event);
    auto ratio = QApplication::desktop()->devicePixelRatio();
    ViewMouseInfo info = { event->type(), event->modifiers(), event->buttons(), ratio * event->pos() };
    emit sig_MouseEvent(info);
    update();
}

void ZOpenGLQuickView::mouseReleaseEvent(QMouseEvent* event)
{
    QQuickView::mouseReleaseEvent(event);
    auto ratio = QApplication::desktop()->devicePixelRatioF();
    ViewMouseInfo info = { event->type(), event->modifiers(), event->buttons(), ratio * event->pos() };
    emit sig_MouseEvent(info);
    update();
}

void ZOpenGLQuickView::mouseMoveEvent(QMouseEvent* event)
{
    QQuickView::mouseMoveEvent(event);
    auto ratio = QApplication::desktop()->devicePixelRatioF();
    ViewMouseInfo info = { event->type(), event->modifiers(), event->buttons(), ratio * event->pos() };
    emit sig_MouseEvent(info);
    update();
}

void ZOpenGLQuickView::wheelEvent(QWheelEvent* event)
{
    QQuickView::wheelEvent(event);
    auto ratio = QApplication::desktop()->devicePixelRatioF();
    ViewMouseInfo info = { event->type(), event->modifiers(), event->buttons(), ratio * event->pos(), event->angleDelta() };
    emit sig_MouseEvent(info);
    update();
}

void ZOpenGLQuickView::resizeEvent(QResizeEvent* event) {
    //mock QGLWidget::resizeEvent
    QQuickView::resizeEvent(event);
    QSize sz = event->size();
    const qreal scaleFactor = devicePixelRatio();
    int nx = sz.width() * scaleFactor, ny = sz.height() * scaleFactor;
    emit sig_Resize(nx, ny);
}

void ZOpenGLQuickView::cleanUpScene() {
    emit sig_cleanScene();
    update();
}

Zenovis* ZOpenGLQuickView::getZenoVis() {
    if (m_renderer)
        return m_renderer->getZenovis();
    else
        return nullptr;
}

void ZOpenGLQuickView::resizeGL(int nx, int ny) {
    emit sig_Resize(nx, ny);
}

void ZOpenGLQuickView::updatePerspective()
{
    emit sig_UpdatePerspective();
}

void ZOpenGLQuickView::setNumSamples(int samples) {
    emit sig_SetNumSamples(samples);
}

void ZOpenGLQuickView::setCameraRes(const QVector2D& res) {
    emit sig_setCameraRes(res);
}

void ZOpenGLQuickView::setSafeFrames(bool bLock, int nx, int ny) {
    emit sig_setSafeFrames(bLock, nx, ny);
}

void ZOpenGLQuickView::setShowPtnum(bool bShow) {
    emit sig_showptnum(bShow);
}

bool ZOpenGLQuickView::isCameraMoving() const {
    return false;
}

void ZOpenGLQuickView::setSimpleRenderOption()
{
    emit sig_setSimpleRenderOption();
}

void ZOpenGLQuickView::clearTransformer()
{
    emit sig_clearTransformer();
}

void ZOpenGLQuickView::changeTransformOperation(const QString& node)
{
    emit sig_changeTransformOperationByNode(node);
}

void ZOpenGLQuickView::changeTransformOperation(int mode)
{
    emit sig_changeTransformOperation(mode);
}

void ZOpenGLQuickView::cameraLookTo(zenovis::CameraLookToDir dir)
{
    emit sig_cameraLookTo(dir);
}

void ZOpenGLQuickView::setViewWidgetInfo(DockContentWidgetInfo& info)
{
    if (m_renderer) {
        m_renderer->setViewWidgetInfo(info);
    }
}
