#include "qdmopenglviewport.h"
#include <QOpenGLBuffer>

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

QSize QDMOpenGLViewport::sizeHint() const
{
    return QSize(768, 640);
}

void QDMOpenGLViewport::initializeGL()
{
    initializeOpenGLFunctions();
    m_vao = std::make_unique<QOpenGLVertexArrayObject>(this);
    m_vao->create();
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, [this] () {
        m_vao->destroy();
        m_vao = nullptr;
    });

    m_program = std::make_unique<QOpenGLShaderProgram>(this);
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
attribute vec3 attrPos;

void main() {
    gl_Position = vec4(attrPos, 1);
}
)");
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
void main() {
    gl_FragColor = vec4(vec3(0.8), 1);
}
)");
    m_program->link();
}

void QDMOpenGLViewport::resizeGL(int nx, int ny)
{

}

void QDMOpenGLViewport::paintGL()
{
    glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio());

    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_vao->bind();
    m_program->bind();

    static const GLfloat vertices[] = {
         0.0f,  0.707f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
    };

    QOpenGLBuffer attrPos;
    attrPos.create();
    attrPos.setUsagePattern(QOpenGLBuffer::StreamDraw);
    attrPos.bind();
    attrPos.allocate(vertices, sizeof(vertices));

    m_program->enableAttributeArray("attrPos");
    m_program->setAttributeBuffer("attrPos", GL_FLOAT, 0, 3);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    m_program->disableAttributeArray("attrPos");
    attrPos.destroy();

    m_program->release();
    m_vao->release();
}
