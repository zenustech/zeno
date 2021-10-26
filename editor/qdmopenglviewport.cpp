#include "qdmopenglviewport.h"
#include "renderable.h"
#include <QOpenGLVertexArrayObject>

QDMOpenGLViewport::QDMOpenGLViewport(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setSamples(8);
    fmt.setVersion(4, 1);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
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
}

void QDMOpenGLViewport::resizeGL(int nx, int ny)
{

}

void QDMOpenGLViewport::paintGL()
{
    glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio());

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QOpenGLVertexArrayObject vao(this);
    vao.bind();
    for (auto const &r: m_renderables) {
        r->render(this);
    }
    vao.release();
}
