#include "qdmopenglviewport.h"
#include "renderable.h"
#include <QOpenGLVertexArrayObject>
#include <QDebug>

ZENO_NAMESPACE_BEGIN

QDMOpenGLViewport::QDMOpenGLViewport(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    //fmt.setSamples(8);
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
    qInfo() << "OpenGL version:" << (char const *)glGetString(GL_VERSION);

    glDisable(GL_DEPTH_TEST);
}

void QDMOpenGLViewport::resizeGL(int nx, int ny)
{
}

void QDMOpenGLViewport::paintGL()
{
    glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio());

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QOpenGLVertexArrayObject vao;
    vao.bind();
    for (auto const &[_, r]: m_renderables) {
        r->render(this);
    }
    vao.release();
}

static std::unique_ptr<Renderable> make_renderable_of_node(QDMGraphicsNode *node) {
    return makeRenderableFromAny(node->getDopNode()->outputs[0]);
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
