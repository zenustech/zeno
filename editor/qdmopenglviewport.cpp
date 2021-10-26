#include "qdmopenglviewport.h"
#include <GL/gl.h>

QDMOpenGLViewport::QDMOpenGLViewport(QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setSamples(8);
    fmt.setVersion(4, 1);
    fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
    setFormat(fmt);
}

QSize QDMOpenGLViewport::sizeHint() const
{
    return QSize(768, 640);
}

void QDMOpenGLViewport::initializeGL()
{
    initializeOpenGLFunctions();

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

    m_program->bind();

    static const GLfloat vertices[] = {
         0.0f,  0.707f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
    };

    auto attrPos = m_program->attributeLocation("attrPos");
    qDebug() << attrPos;
    Q_ASSERT(attrPos != -1);
    glEnableVertexAttribArray(attrPos);
    glVertexAttribPointer(attrPos, 3, GL_FLOAT, GL_FALSE, 0, vertices);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisableVertexAttribArray(attrPos);

    m_program->release();
}
