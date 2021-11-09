#include "qdmopenglviewport.h"
#include "renderable.h"
#include <QOpenGLVertexArrayObject>
#include <QDragMoveEvent>
#include <QWheelEvent>
#include <zeno/zmt/log.h>
#include <zeno/dop/execute.h>
#include <zeno/dop/Exception.h>

ZENO_NAMESPACE_BEGIN

QDMOpenGLViewport::QDMOpenGLViewport(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setSamples(8);
    //fmt.setVersion(3, 1);
    setFormat(fmt);
}

QDMOpenGLViewport::~QDMOpenGLViewport() = default;

QSize QDMOpenGLViewport::sizeHint() const
{
    return QSize(768, 640);
}

void QDMOpenGLViewport::initializeGL()
{
    initializeOpenGLFunctions();
    ZENO_LOG_INFO("OpenGL version: {}", (char const *)glGetString(GL_VERSION));

    glEnable(GL_DEPTH_TEST);
}

void QDMOpenGLViewport::resizeGL(int nx, int ny)
{
}

CameraData *QDMOpenGLViewport::getCamera() const
{
    return m_camera.get();
}

void QDMOpenGLViewport::paintGL()
{
    int nx = width() * devicePixelRatio();
    int ny = height() * devicePixelRatio();
    m_camera->resize(nx, ny);
    glViewport(0, 0, nx, ny);

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QOpenGLVertexArrayObject vao;
    vao.bind();
    for (auto const &[_, r]: m_renderables) {
        r->render(this);
    }
    vao.release();
}

void QDMOpenGLViewport::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        m_mmbPos = event->pos();
    }

    QOpenGLWidget::mousePressEvent(event);
}

void QDMOpenGLViewport::mouseMoveEvent(QMouseEvent *event)
{
    if (m_mmbPos) {
        auto delta = event->pos() - *m_mmbPos;
        m_camera->move((float)delta.x() / width(), -(float)delta.y() / height(),
                       event->modifiers() & Qt::ShiftModifier);
        m_mmbPos = event->pos();
        repaint();
    }

    QOpenGLWidget::mousePressEvent(event);
}

void QDMOpenGLViewport::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton) {
        m_mmbPos = std::nullopt;
    }

    QOpenGLWidget::mousePressEvent(event);
}

void QDMOpenGLViewport::wheelEvent(QWheelEvent *event)
{
    float dy = event->angleDelta().y();
    if (dy > 0) dy = 1.f;
    if (dy < 0) dy = -1.f;
    m_camera->zoom(dy, event->modifiers() & Qt::ShiftModifier);
    repaint();

    QOpenGLWidget::wheelEvent(event);
}

static std::unique_ptr<Renderable> make_renderable_of_node(QDMGraphicsNode *node) {
    auto dopNode = node->getDopNode();
    ztd::any_ptr val;
    try {
        val = dop::resolve(dop::Input_Link{.node = dopNode, .sockid = 0});
    } catch (dop::Exception const &e) {
        ZENO_LOG_INFO("Exception during DOP execution: {}", e.what());
    }
    return makeRenderableFromAny(val);
}

void QDMOpenGLViewport::updateNode(QDMGraphicsNode *node, int type) {
    if (type > 0) {
        m_renderables.emplace(node, make_renderable_of_node(node));
    } else if (type == 0) {
        m_renderables.at(node) = make_renderable_of_node(node);
    } else {
        m_renderables.erase(node);
    }
    repaint();
}

ZENO_NAMESPACE_END
